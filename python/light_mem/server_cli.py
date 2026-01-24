import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import signal
import time
from textwrap import dedent


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(cmd: list[str]) -> int:
    try:
        p = subprocess.run(cmd, check=False)
        return int(p.returncode)
    except FileNotFoundError:
        return 127


def _docker_compose_cmd() -> list[str] | None:
    # Prefer: docker compose
    if _which("docker"):
        # Best-effort: if `docker compose version` works, use it.
        rc = _run(["docker", "compose", "version"])
        if rc == 0:
            return ["docker", "compose"]

    # Fallback: legacy docker-compose
    if _which("docker-compose"):
        return ["docker-compose"]

    return None


def _stop_docker_services(*, purge_volumes: bool) -> int:
    docker = _which("docker")
    if not docker:
        sys.stderr.write("docker is required for lightmem_server --stop.\n")
        return 2

    # Containers are created with fixed names.
    containers = ["lightmem-index", "lightmem-coord"]
    rm = subprocess.run([docker, "rm", "-f", *containers], capture_output=True, text=True, check=False)

    # If containers do not exist, treat as success.
    if rm.returncode != 0:
        err = (rm.stderr or "") + (rm.stdout or "")
        lowered = err.lower()
        if "no such container" not in lowered:
            sys.stderr.write(err)
            sys.stderr.write("Failed to stop containers.\n")
            return int(rm.returncode)

    if purge_volumes:
        # Volumes are named explicitly by our compose YAML.
        vols = ["lightmem-redis-data", "lightmem-etcd-data"]
        vrm = subprocess.run([docker, "volume", "rm", "-f", *vols], capture_output=True, text=True, check=False)
        if vrm.returncode != 0:
            err = (vrm.stderr or "") + (vrm.stdout or "")
            lowered = err.lower()
            # Ignore missing volumes.
            if "no such volume" not in lowered:
                sys.stderr.write(err)
                sys.stderr.write("Failed to remove volumes.\n")
                return int(vrm.returncode)

    sys.stdout.write("lightmem_server stopped docker services.\n")
    if purge_volumes:
        sys.stdout.write("Removed volumes: lightmem-redis-data, lightmem-etcd-data\n")
    return 0


def _port_open(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _redis_ping(host: str, port: int) -> bool:
    import socket

    payload = "*1\r\n$4\r\nPING\r\n".encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=0.5) as s:
            s.settimeout(0.5)
            s.sendall(payload)
            data = s.recv(64)
        return data.startswith(b"+PONG")
    except OSError:
        return False


def _etcd_health(host: str, port: int) -> bool:
    import json
    import urllib.request

    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=0.8) as resp:
            raw = resp.read()
        obj = json.loads(raw.decode("utf-8")) if raw else {}
        # etcd returns {"health":"true"}
        return str(obj.get("health", "")).lower() == "true"
    except Exception:
        return False


def _detect_advertise_host() -> str | None:
    """Best-effort detect a non-loopback IPv4 address for cross-host access."""
    import socket

    # Common technique: create a UDP socket to a public IP; no packets need to be sent.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Fallback: resolve hostname.
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return None


def _wait_port(host: str, port: int, *, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if _port_open(host, port):
            return True
        time.sleep(0.1)
    return False


def _start_redis_local(*, host: str, port: int, data_dir: str) -> subprocess.Popen:
    redis_server = _which("redis-server")
    if not redis_server:
        raise RuntimeError("redis-server not found in PATH")

    if _port_open(host, port):
        raise RuntimeError(f"redis port already in use: {host}:{port}")

    cmd = [
        redis_server,
        "--bind",
        host,
        "--port",
        str(port),
        "--save",
        "",
        "--appendonly",
        "yes",
        "--appendfsync",
        "everysec",
        "--dir",
        data_dir,
    ]
    proc = subprocess.Popen(cmd)
    if not _wait_port(host, port, timeout_s=10.0):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError("redis-server failed to start")
    return proc


def _start_etcd_local(*, host: str, client_port: int, peer_port: int, data_dir: str) -> subprocess.Popen:
    etcd = _which("etcd")
    if not etcd:
        raise RuntimeError("etcd not found in PATH")

    if _port_open(host, client_port):
        raise RuntimeError(f"etcd client port already in use: {host}:{client_port}")
    if _port_open(host, peer_port):
        raise RuntimeError(f"etcd peer port already in use: {host}:{peer_port}")

    name = "coord"
    cmd = [
        etcd,
        "--name",
        name,
        "--data-dir",
        data_dir,
        "--listen-client-urls",
        f"http://{host}:{client_port}",
        "--advertise-client-urls",
        f"http://{host}:{client_port}",
        "--listen-peer-urls",
        f"http://{host}:{peer_port}",
        "--initial-advertise-peer-urls",
        f"http://{host}:{peer_port}",
        "--initial-cluster",
        f"{name}=http://{host}:{peer_port}",
        "--initial-cluster-state",
        "new",
    ]
    proc = subprocess.Popen(cmd)
    if not _wait_port(host, client_port, timeout_s=15.0):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError("etcd failed to start")
    return proc


def _run_local_services(*, index_port: int, coord_port: int, coord_peer_port: int) -> int:
    host = "127.0.0.1"
    with tempfile.TemporaryDirectory(prefix="lightmem_server_local_") as td:
        redis_dir = os.path.join(td, "redis")
        etcd_dir = os.path.join(td, "etcd")
        os.makedirs(redis_dir, exist_ok=True)
        os.makedirs(etcd_dir, exist_ok=True)

        redis_proc: subprocess.Popen | None = None
        etcd_proc: subprocess.Popen | None = None
        manage_redis = True
        manage_etcd = True

        def _stop_all() -> None:
            nonlocal redis_proc, etcd_proc
            for p in (etcd_proc, redis_proc):
                if p is None:
                    continue
                try:
                    p.terminate()
                except Exception:
                    pass
            for p in (etcd_proc, redis_proc):
                if p is None:
                    continue
                try:
                    p.wait(timeout=8)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

        def _sig_handler(_signum, _frame) -> None:  # pragma: no cover
            _stop_all()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        try:
            # If ports are already occupied, assume external services and reuse them.
            if _port_open(host, index_port):
                manage_redis = False
                if not _redis_ping(host, index_port):
                    raise RuntimeError(f"port {host}:{index_port} is in use but does not look like Redis")
            else:
                redis_proc = _start_redis_local(host=host, port=index_port, data_dir=redis_dir)

            if _port_open(host, coord_port):
                manage_etcd = False
                if not _etcd_health(host, coord_port):
                    raise RuntimeError(f"port {host}:{coord_port} is in use but does not look like etcd")
            else:
                etcd_proc = _start_etcd_local(
                    host=host,
                    client_port=coord_port,
                    peer_port=coord_peer_port,
                    data_dir=etcd_dir,
                )
        except Exception as e:
            if manage_etcd or manage_redis:
                _stop_all()
            sys.stderr.write(f"Failed to start local services: {e}\n")
            sys.stderr.write("Hint: install redis-server and etcd, or use --mode docker with docker compose.\n")
            return 2

        sys.stdout.write("lightmem_server local services ready (foreground).\n")
        if not manage_redis:
            sys.stdout.write("Index: (reused existing)\n")
        if not manage_etcd:
            sys.stdout.write("Coord:  (reused existing)\n")
        sys.stdout.write(f"Index: {host}:{index_port}\n")
        sys.stdout.write(f"Coord:  {host}:{coord_port}\n")
        sys.stdout.write("Press Ctrl-C to stop.\n\n")

        # Block until a managed child exits (if we started any).
        try:
            while True:
                if manage_redis and redis_proc is not None and redis_proc.poll() is not None:
                    sys.stderr.write("redis-server exited; stopping...\n")
                    break
                if manage_etcd and etcd_proc is not None and etcd_proc.poll() is not None:
                    sys.stderr.write("etcd exited; stopping...\n")
                    break
                time.sleep(0.2)
        finally:
            # Only stop what we started.
            if manage_etcd or manage_redis:
                _stop_all()
        return 0


def _compose_yaml(*, redis_port: int, etcd_client_port: int, etcd_peer_port: int) -> str:
        # Minimal single-node index backend + coordinator backend.
        # This is intentionally a thin wrapper that starts dependency services.
        return dedent(
                f"""
                services:
                    index:
                        image: redis:7-alpine
                        container_name: lightmem-index
                        ports:
                            - "{redis_port}:6379"
                        command: ["redis-server", "--save", "", "--appendonly", "yes", "--appendfsync", "everysec"]
                        volumes:
                            - redis-data:/data
                        restart: unless-stopped

                    coord:
                        image: quay.io/coreos/etcd:v3.5.12
                        container_name: lightmem-coord
                        environment:
                            - ETCD_NAME=coord
                            - ETCD_DATA_DIR=/etcd-data
                            - ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379
                            - ETCD_ADVERTISE_CLIENT_URLS=http://coord:2379
                            - ETCD_LISTEN_PEER_URLS=http://0.0.0.0:2380
                            - ETCD_INITIAL_ADVERTISE_PEER_URLS=http://coord:2380
                            - ETCD_INITIAL_CLUSTER=coord=http://coord:2380
                            - ETCD_INITIAL_CLUSTER_STATE=new
                            - ETCD_INITIAL_CLUSTER_TOKEN=lightmem
                        ports:
                            - "{etcd_client_port}:2379"
                            - "{etcd_peer_port}:2380"
                        volumes:
                            - etcd-data:/etcd-data
                        restart: unless-stopped

                volumes:
                    redis-data:
                        name: lightmem-redis-data
                    etcd-data:
                        name: lightmem-etcd-data
                """
        ).lstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lightmem_server",
        description="Start LightMem dependency services (index backend + coordinator backend) via Docker Compose.",
    )
    parser.add_argument("--index-port", type=int, default=6379, help="Host port for the index backend (mapped to 6379)")
    parser.add_argument("--coord-port", type=int, default=2379, help="Host port for the coordinator client (mapped to 2379)")
    parser.add_argument("--coord-peer-port", type=int, default=2380, help="Host port for the coordinator peer (mapped to 2380)")
    parser.add_argument(
        "--mode",
        choices=["auto", "docker", "local"],
        default="auto",
        help="Start mode: docker (docker compose), local (redis-server+etcd), or auto.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop docker-mode services (removes containers lightmem-index/lightmem-coord).",
    )
    parser.add_argument(
        "--purge-volumes",
        action="store_true",
        help="With --stop: also remove named volumes (lightmem-redis-data/lightmem-etcd-data).",
    )
    parser.add_argument(
        "--coord-ttl",
        type=int,
        default=None,
        help="Coordinator TTL seconds (passed into PyLocalCacheService example)",
    )

    parser.add_argument(
        "--coord-reconcile-sec",
        type=float,
        default=None,
        help="Coordinator fallback reconcile interval seconds (passed into PyLocalCacheService example)",
    )

    parser.add_argument(
        "--advertise-host",
        default="",
        help="Host/IP to print in endpoints for cross-host clients. Empty means auto-detect (docker) or 127.0.0.1 (local).",
    )

    args = parser.parse_args(argv)

    if args.purge_volumes and not args.stop:
        sys.stderr.write("--purge-volumes requires --stop.\n")
        return 2

    if args.stop:
        return _stop_docker_services(purge_volumes=bool(args.purge_volumes))

    # The services started by this CLI may be reachable from other machines (docker mode).
    # This only affects printed endpoints/snippets; it does not change bind behavior.
    if args.mode == "local":
        advertise_host = "127.0.0.1"
    else:
        advertise_host = str(args.advertise_host).strip() or (_detect_advertise_host() or "127.0.0.1")

    if args.mode == "local":
        return _run_local_services(index_port=int(args.index_port), coord_port=int(args.coord_port), coord_peer_port=int(args.coord_peer_port))

    compose = _docker_compose_cmd()
    if args.mode in ("auto", "docker") and compose:
        pass
    elif args.mode == "docker":
        sys.stderr.write("docker compose is required for lightmem_server --mode docker.\n")
        sys.stderr.write("On Ubuntu: install docker-compose-plugin or docker-compose.\n")
        return 2
    else:
        # auto fallback
        return _run_local_services(index_port=int(args.index_port), coord_port=int(args.coord_port), coord_peer_port=int(args.coord_peer_port))

    if not (1 <= args.index_port <= 65535):
        sys.stderr.write("Invalid --index-port.\n")
        return 2
    if not (1 <= args.coord_port <= 65535):
        sys.stderr.write("Invalid --coord-port.\n")
        return 2
    if not (1 <= args.coord_peer_port <= 65535):
        sys.stderr.write("Invalid --coord-peer-port.\n")
        return 2

    if args.coord_port == args.coord_peer_port:
        sys.stderr.write("--coord-port and --coord-peer-port must differ.\n")
        return 2

    if args.coord_ttl is not None and args.coord_ttl <= 0:
        sys.stderr.write("Invalid --coord-ttl (must be > 0).\n")
        return 2
    if args.coord_reconcile_sec is not None and args.coord_reconcile_sec <= 0:
        sys.stderr.write("Invalid --coord-reconcile-sec (must be > 0).\n")
        return 2


    yml = _compose_yaml(redis_port=args.index_port, etcd_client_port=args.coord_port, etcd_peer_port=args.coord_peer_port)

    with tempfile.TemporaryDirectory(prefix="lightmem_server_") as td:
        path = os.path.join(td, "docker-compose.lightmem-server.yml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yml)

        # Use a stable project name: we use stable container/volume names and a
        # temp compose file path, so the default derived project name would vary
        # per run and cause warnings about existing volumes.
        cmd = [*compose, "--project-name", "lightmem_server", "-f", path, "up", "-d"]
        rc = _run(cmd)
        if rc != 0:
            sys.stderr.write("Failed to start services via Docker Compose.\n")
            return rc

    # Print minimal connection info for users.
    sys.stdout.write("lightmem_server started dependency services.\n")
    sys.stdout.write(f"Index: {advertise_host}:{args.index_port}\n")
    sys.stdout.write(f"Coord:  {advertise_host}:{args.coord_port}\n")
    sys.stdout.write("\nExample Python for LightMem clients:\n")
    sys.stdout.write("  from light_mem import PyLocalCacheService\n")
    sys.stdout.write("  # ... prepare kvcache_tensor, file, etc ...\n")
    sys.stdout.write("  svc = PyLocalCacheService(\n")
    sys.stdout.write("      kvcache_tensor=kvcache_tensor,\n")
    sys.stdout.write("      file=file,\n")
    if args.index_port == 6379 and args.coord_port == 2379:
        sys.stdout.write(f"      index_endpoint=\"{advertise_host}\",\n")
    else:
        sys.stdout.write(f"      index_endpoint=\"{advertise_host}:{args.index_port}\",\n")
        sys.stdout.write(f"      coord_endpoints=\"{advertise_host}:{args.coord_port}\",\n")
    if args.coord_ttl is not None:
        sys.stdout.write(f"      coord_ttl={int(args.coord_ttl)},\n")
    if args.coord_reconcile_sec is not None:
        sys.stdout.write(f"      coord_reconcile_sec={float(args.coord_reconcile_sec)},\n")
    sys.stdout.write("  )\n")
    sys.stdout.write("\n")
    return 0
