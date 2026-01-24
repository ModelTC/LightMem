import threading
import time
import socket
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .etcd_v3_http import EtcdV3HttpClient


@dataclass(frozen=True)
class EtcdOptions:
    endpoints: str
    prefix: str
    node_id: str
    ttl: int
    reconcile_interval_sec: float


def _parse_endpoints(endpoints: str) -> Tuple[str, int]:
    # We pick the first endpoint.
    first = (endpoints or "").split(",")[0].strip()
    if not first:
        return "127.0.0.1", 2379
    if ":" in first:
        host, port = first.rsplit(":", 1)
        try:
            return host, int(port)
        except Exception:
            return host, 2379
    return first, 2379


def _hrw_score(node_id: str, shard_id: int) -> int:
    # Deterministic 64-bit score for HRW / Rendezvous hashing.
    #
    # Note: avoid simplistic rolling hashes here; for structured inputs like
    # "node-{i}:{sid}" they can exhibit pathological ordering (one node wins
    # almost all shards). We use stdlib blake2b to get a stable, well-distributed
    # 64-bit score without external dependencies.
    h = hashlib.blake2b(digest_size=8)
    h.update(node_id.encode("utf-8"))
    h.update(b":")
    h.update(int(shard_id).to_bytes(4, byteorder="big", signed=False))
    return int.from_bytes(h.digest(), byteorder="big", signed=False)


class EtcdShardCoordinator(threading.Thread):
    """Drive shard ownership via etcd and feed assignments into the C++ engine.

    This runs on Python side to avoid adding heavy C++ dependencies (gRPC/protobuf).

    Keys (under prefix):
      - nodes/{node_id}              (lease-bound)
      - shards/{sid}/state           (FREE|CLAIMED|DRAINING)
      - shards/{sid}/owner           (node_id, lease-bound)
      - shards/{sid}/handoff_to      (node_id)  # request old owner to drain/release

    Epoch fencing:
            - epoch = create_revision(owner_key) after a successful claim
      - passed down into C++ via update_shard_assignments()
      - C++ persists epoch into shard superblock + writes an epoch marker to WAL

    Delayed handoff:
      - old owner sets draining=1 locally when it sees handoff_to != self or desired owner changed
      - coordinator waits until C++ reports inflight==0 (writes + online local-hit reads) before deleting owner key
    """

    def __init__(self, service, num_shards: int, opt: EtcdOptions):
        super().__init__(daemon=True)
        self._svc = service
        self._num_shards = int(num_shards)
        self._opt = opt

        self._prefix = (opt.prefix or "lightmem").rstrip("/")

        self._stop_evt = threading.Event()

        # Event-driven reconcile: etcd watch callbacks set this.
        self._dirty_evt = threading.Event()
        self._watch_ids: List[int] = []
        self._watch_enabled = False
        self._watch_debounce_sec: float = 0.2

        self._owned_epoch: Dict[int, int] = {}
        self._draining: Dict[int, bool] = {}

        self._last_nodes: List[str] = []
        self._assignment: Dict[int, str] = {}


    def stop(self) -> None:
        self._stop_evt.set()
        self._dirty_evt.set()

    def _client(self):
        host, port = _parse_endpoints(self._opt.endpoints)
        return EtcdV3HttpClient(host=host, port=port)

    def _k(self, suffix: str) -> str:
        # Normalize to avoid accidental double slashes.
        s = (suffix or "").lstrip("/")
        return f"{self._prefix}/{s}".rstrip("/")

    def _nodes_prefix(self) -> str:
        return f"{self._prefix}/nodes/"

    def _shards_prefix(self) -> str:
        return f"{self._prefix}/shards/"

    def _ensure_state(self, client, sid: int) -> None:
        state_key = self._k(f"shards/{sid}/state")
        # Create default state FREE if absent
        client.transaction(
            compare=[client.transactions.create(state_key) == 0],
            success=[client.transactions.put(state_key, "FREE")],
            failure=[],
        )

    def _get_nodes(self, client) -> List[str]:
        prefix = self._nodes_prefix()
        nodes: List[str] = []
        for _, meta in client.get_prefix(prefix):
            try:
                key = meta.key.decode("utf-8")
            except Exception:
                continue
            if not key.startswith(prefix):
                continue
            node_id = key[len(prefix):]
            if node_id:
                nodes.append(node_id)
        nodes.sort()
        return nodes

    def _start_watchers(self, client) -> None:
        # Best-effort: if watch APIs are unavailable, we fall back to polling.
        self._watch_ids.clear()
        self._watch_enabled = False

        add_watch = getattr(client, "add_watch_prefix_callback", None)
        if not callable(add_watch):
            return

        def _on_evt(*_args, **_kwargs) -> None:
            # Coalesce bursts.
            self._dirty_evt.set()

        try:
            wid1 = add_watch(self._nodes_prefix(), _on_evt)
            wid2 = add_watch(self._shards_prefix(), _on_evt)
            for wid in (wid1, wid2):
                if isinstance(wid, int):
                    self._watch_ids.append(wid)
            self._watch_enabled = True
        except Exception:
            self._watch_ids.clear()
            self._watch_enabled = False

    def _stop_watchers(self, client) -> None:
        cancel = getattr(client, "cancel_watch", None)
        if callable(cancel):
            for wid in list(self._watch_ids):
                try:
                    cancel(wid)
                except Exception:
                    pass
        self._watch_ids.clear()
        self._watch_enabled = False

    def _reconcile_once(self, client, lease) -> None:
        nodes = self._get_nodes(client)

        # Recompute assignment only when node membership changes.
        self._recompute_assignment_on_membership_change(client, nodes)

        shard_ids: List[int] = []
        epochs: List[int] = []
        draining: List[int] = []

        newly_claimed: List[int] = []

        for sid in range(self._num_shards):
            desired = self._assignment.get(sid)

            state_key = self._k(f"shards/{sid}/state")
            owner_key = self._k(f"shards/{sid}/owner")
            handoff_key = self._k(f"shards/{sid}/handoff_to")

            state = self._get_text(client, state_key)
            if state is None:
                # Lazy init to reduce etcd traffic.
                self._ensure_state(client, sid)
                state = self._get_text(client, state_key)
            state = state or "FREE"

            owner = self._get_text(client, owner_key)
            handoff_to = self._get_text(client, handoff_key)

            if desired == self._opt.node_id:
                if owner is None:
                    ep = self._claim(client, sid, lease)
                    if ep:
                        if sid not in self._owned_epoch:
                            newly_claimed.append(sid)
                        self._owned_epoch[sid] = ep
                        self._draining[sid] = False
                elif owner != self._opt.node_id:
                    # Ask current owner to drain/release.
                    self._request_handoff(client, sid)
                else:
                    # We own it.
                    v, meta = client.get(owner_key)
                    # IMPORTANT: use create_revision as epoch fencing.
                    # mod_revision can change on lease refresh (PUT with same value), which would
                    # otherwise cause epoch churn and heavy metadata write contention in C++.
                    ep = int(getattr(meta, "create_revision", 0) or 0)
                    if not ep:
                        ep = int(getattr(meta, "mod_revision", 0) or 0)
                    if ep:
                        self._owned_epoch[sid] = ep
                    # If someone else requested handoff, start draining.
                    # If this node is the desired owner, clear stale handoff_to to avoid
                    # permanently-drained shards after transient membership changes.
                    if handoff_to and handoff_to != self._opt.node_id:
                        if desired == self._opt.node_id:
                            try:
                                client.delete(handoff_key)
                            except Exception:
                                pass
                            self._draining[sid] = False
                        else:
                            self._draining[sid] = True
                    else:
                        self._draining[sid] = False
            else:
                if owner == self._opt.node_id:
                    # We should no longer own it; start draining and release when safe.
                    self._draining[sid] = True
                    self._release_if_safe(client, sid, desired)
                else:
                    self._owned_epoch.pop(sid, None)
                    self._draining.pop(sid, None)

        # Push assignments into C++.
        for sid, ep in self._owned_epoch.items():
            shard_ids.append(int(sid))
            epochs.append(int(ep))
            draining.append(1 if self._draining.get(sid, False) else 0)

        self._svc.update_shard_assignments(shard_ids, epochs, draining)

        # On newly claimed shards, recover Redis index with a smart policy.
        for sid in newly_claimed:
            try:
                self._svc.recover_shard_to_redis_smart(int(sid))
            except Exception:
                pass

    def _desired_owner(self, nodes: List[str], sid: int) -> Optional[str]:
        if not nodes:
            return None
        best_node = None
        best_score = None
        for n in nodes:
            sc = _hrw_score(n, sid)
            if best_score is None or sc > best_score:
                best_score = sc
                best_node = n
        return best_node

    def _recompute_assignment_on_membership_change(self, client, nodes: List[str]) -> None:
        if nodes == self._last_nodes and self._assignment:
            return

        self._last_nodes = list(nodes)
        self._assignment.clear()
        if not nodes:
            return

        # HRW / Rendezvous hashing assignment:
        # - Deterministic mapping shard -> node based solely on (node_id, shard_id)
        # - Minimal movement when nodes join/leave
        # - Stable when membership unchanged
        for sid in range(self._num_shards):
            self._assignment[sid] = self._desired_owner(nodes, sid)  # type: ignore[assignment]

    def _get_text(self, client, key: str) -> Optional[str]:
        v, _ = client.get(key)
        if v is None:
            return None
        try:
            return v.decode("utf-8")
        except Exception:
            return None

    def _claim(self, client, sid: int, lease) -> Optional[int]:
        owner_key = self._k(f"shards/{sid}/owner")
        state_key = self._k(f"shards/{sid}/state")
        handoff_key = self._k(f"shards/{sid}/handoff_to")

        # Only claim if no owner.
        txn_ok, _ = client.transaction(
            compare=[
                client.transactions.create(owner_key) == 0,
            ],
            success=[
                client.transactions.put(owner_key, self._opt.node_id, lease=lease),
                client.transactions.put(state_key, "CLAIMED"),
                client.transactions.delete(handoff_key),
            ],
            failure=[],
        )
        if not txn_ok:
            return None

        # epoch = create_revision(owner_key) (stable across lease refresh)
        v, meta = client.get(owner_key)
        if v is None or meta is None:
            return None
        ep = int(getattr(meta, "create_revision", 0) or 0)
        if not ep:
            ep = int(getattr(meta, "mod_revision", 0) or 0)
        return ep

    def _request_handoff(self, client, sid: int) -> None:
        handoff_key = self._k(f"shards/{sid}/handoff_to")
        # Always overwrite to avoid stale handoff targets pinning shards in draining state.
        try:
            client.put(handoff_key, self._opt.node_id)
        except Exception:
            # Best-effort; next reconcile will retry.
            pass

    def _release_if_safe(self, client, sid: int, desired_owner: Optional[str]) -> None:
        owner_key = self._k(f"shards/{sid}/owner")
        state_key = self._k(f"shards/{sid}/state")

        # Only release if no inflight operations (writes + online local-hit reads).
        inflight = int(self._svc.shard_inflight(int(sid)))
        if inflight != 0:
            return

        # Best-effort state update + owner delete.
        try:
            if desired_owner and desired_owner != self._opt.node_id:
                client.put(state_key, "FREE")
            client.delete(owner_key)
            self._owned_epoch.pop(sid, None)
            self._draining.pop(sid, None)
        except Exception:
            return

    def run(self) -> None:
        def _best_effort_cleanup(c) -> None:
            # Best-effort cleanup to reduce stale membership/ownership during
            # process restarts (e.g., pipeline phases). This is intentionally
            # conservative: only delete owner keys if they still point to us.
            try:
                node_key_local = self._k(f"nodes/{self._opt.node_id}")
                try:
                    c.delete(node_key_local)
                except Exception:
                    pass

                for sid in range(int(self._num_shards)):
                    owner_key = self._k(f"shards/{int(sid)}/owner")
                    state_key = self._k(f"shards/{int(sid)}/state")
                    try:
                        v, _m = c.get(owner_key)
                        owner = None
                        if v is not None:
                            try:
                                owner = v.decode("utf-8")
                            except Exception:
                                owner = None
                        if owner == self._opt.node_id:
                            try:
                                c.put(state_key, "FREE")
                            except Exception:
                                pass
                            try:
                                c.delete(owner_key)
                            except Exception:
                                pass
                    except Exception:
                        continue
            except Exception:
                return

        client = self._client()
        lease = client.lease(self._opt.ttl)

        # Register node key bound to the lease (one-time); keepalive refreshes the lease.
        node_key = self._k(f"nodes/{self._opt.node_id}")
        client.put(node_key, "1", lease=lease)

        self._start_watchers(client)
        # Force an initial reconcile.
        self._dirty_evt.set()

        keepalive_interval = max(1.0, float(self._opt.ttl) / 3.0)
        fallback_reconcile_interval = max(1.0, float(self._opt.reconcile_interval_sec))

        next_keepalive = 0.0
        next_fallback_reconcile = 0.0

        try:
            while not self._stop_evt.is_set():
                now = time.time()

                # Keep the lease alive.
                if now >= next_keepalive:
                    try:
                        # Use one-shot keepalive against the grpc-gateway streaming endpoint.
                        lease = client.lease_keepalive_once(lease)
                    except Exception:
                        # Re-establish client/lease/watch on failures.
                        try:
                            self._stop_watchers(client)
                        except Exception:
                            pass
                        client = self._client()
                        lease = client.lease(self._opt.ttl)
                        client.put(node_key, "1", lease=lease)
                        # Re-attach owned shard owner keys to the new lease.
                        for sid in list(self._owned_epoch.keys()):
                            owner_key = self._k(f"shards/{int(sid)}/owner")
                            try:
                                client.put(owner_key, self._opt.node_id, lease=lease)
                            except Exception:
                                pass
                        self._start_watchers(client)
                        self._dirty_evt.set()
                    next_keepalive = now + keepalive_interval

                # Decide how long to wait for events.
                timeout = max(0.0, min(next_keepalive - now, next_fallback_reconcile - now))
                if self._dirty_evt.is_set():
                    timeout = 0.0

                fired = self._dirty_evt.wait(timeout=timeout)
                if self._stop_evt.is_set():
                    break

                # Debounce watch bursts.
                if fired:
                    self._dirty_evt.clear()
                    time.sleep(self._watch_debounce_sec)

                now = time.time()
                should_reconcile = fired or (now >= next_fallback_reconcile)
                if not should_reconcile:
                    continue

                try:
                    self._reconcile_once(client, lease)
                except Exception:
                    # Fail-closed on coordination errors: stop issuing new writes until we can reconcile again.
                    try:
                        self._svc.update_shard_assignments([], [], [])
                    except Exception:
                        pass

                next_fallback_reconcile = time.time() + fallback_reconcile_interval
        finally:
            try:
                self._stop_watchers(client)
            except Exception:
                pass
            _best_effort_cleanup(client)

# Cluster-level guard: ensure all nodes agree on shard ID space size.
#
# Without this, different nodes may operate on different shard id ranges,
# leading to inconsistent ownership keys and unsafe writes.
def _ensure_cluster_num_shards(endpoints: str, prefix: str, expected: int) -> None:
    host, port = _parse_endpoints(endpoints)
    client = EtcdV3HttpClient(host=host, port=port)

    key = f"{prefix.rstrip('/')}/config/num_shards"

    # First node wins by creating the key. Others validate it matches.
    try:
        ok, _ = client.transaction(
            compare=[client.transactions.create(key) == 0],
            success=[client.transactions.put(key, str(int(expected)))],
            failure=[],
        )
    except Exception as e:
        raise RuntimeError(f"etcd transaction failed while checking {key}: {e}") from e

    if ok:
        return

    v, _ = client.get(key)
    if v is None:
        # Extremely unlikely (key existed for compare but missing now); treat as misconfiguration.
        raise RuntimeError(f"cluster shard config key disappeared: {key}")
    try:
        actual = int(v.decode("utf-8").strip())
    except Exception:
        raise RuntimeError(f"cluster shard config key is not an int: {key}={v!r}")

    if int(actual) != int(expected):
        raise RuntimeError(
            f"num_shards mismatch across nodes: expected {int(expected)} but etcd has {int(actual)} at {key}"
        )

def maybe_start_etcd_coordinator(
    service,
    num_shards: int,
    *,
    endpoints: str,
    index_prefix: str,
    node_id: Optional[str],
    coord_ttl: int,
    coord_reconcile_sec: float,
) -> Optional[EtcdShardCoordinator]:
    endpoints = (endpoints or "").strip()
    if not endpoints:
        return None

    node_id_final = (node_id or "").strip() or (socket.gethostname() or "unknown").strip()
    prefix = (index_prefix or "").strip()
    if not prefix:
        raise ValueError("index_prefix must be non-empty when etcd coordination is enabled")

    ttl = int(coord_ttl)
    interval = float(coord_reconcile_sec)

    _ensure_cluster_num_shards(endpoints=endpoints, prefix=prefix, expected=int(num_shards))

    opt = EtcdOptions(
        endpoints=endpoints,
        prefix=prefix,
        node_id=node_id_final,
        ttl=int(ttl),
        reconcile_interval_sec=float(interval),
    )
    t = EtcdShardCoordinator(service=service, num_shards=int(num_shards), opt=opt)
    t.start()
    return t
