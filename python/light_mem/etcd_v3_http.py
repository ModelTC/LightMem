from __future__ import annotations

import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, Dict, Callable


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _prefix_range_end(prefix: bytes) -> bytes:
    # Standard etcd prefix end calculation: increment the last non-0xFF byte.
    if not prefix:
        return b"\0"
    ba = bytearray(prefix)
    for i in range(len(ba) - 1, -1, -1):
        if ba[i] != 0xFF:
            ba[i] += 1
            return bytes(ba[: i + 1])
    return b"\0"


@dataclass(frozen=True)
class EtcdMeta:
    key: bytes
    create_revision: int
    mod_revision: int
    version: int


class _CreateCompareBuilder:
    def __init__(self, key: str):
        self._key = key

    def __eq__(self, other: object):
        # create(key) == 0  -> VERSION(key) == 0
        try:
            v = int(other)  # type: ignore[arg-type]
        except Exception:
            v = 0
        return _TxnCompare.version_equal(self._key, v)


class _ValueCompareBuilder:
    def __init__(self, key: str):
        self._key = key

    def __eq__(self, other: object):
        b = other if isinstance(other, (bytes, bytearray)) else str(other).encode("utf-8")
        return _TxnCompare.value_equal(self._key, bytes(b))

    def __ne__(self, other: object):
        b = other if isinstance(other, (bytes, bytearray)) else str(other).encode("utf-8")
        return _TxnCompare.value_not_equal(self._key, bytes(b))


@dataclass(frozen=True)
class _TxnCompare:
    target: str
    key: str
    result: str
    version: Optional[int] = None
    value: Optional[bytes] = None

    @staticmethod
    def version_equal(key: str, version: int) -> "_TxnCompare":
        return _TxnCompare(target="VERSION", key=key, result="EQUAL", version=int(version))

    @staticmethod
    def value_equal(key: str, value: bytes) -> "_TxnCompare":
        return _TxnCompare(target="VALUE", key=key, result="EQUAL", value=value)

    @staticmethod
    def value_not_equal(key: str, value: bytes) -> "_TxnCompare":
        return _TxnCompare(target="VALUE", key=key, result="NOT_EQUAL", value=value)


@dataclass(frozen=True)
class _TxnPut:
    key: str
    value: str
    lease_id: int = 0


@dataclass(frozen=True)
class _TxnDelete:
    key: str


class _TxnOps:
    def create(self, key: str) -> _CreateCompareBuilder:
        return _CreateCompareBuilder(key)

    def value(self, key: str) -> _ValueCompareBuilder:
        return _ValueCompareBuilder(key)

    def put(self, key: str, value: str, *, lease=None) -> _TxnPut:
        lease_id = 0
        if lease is not None:
            lease_id = int(getattr(lease, "id", lease) or 0)
        return _TxnPut(key=key, value=str(value), lease_id=lease_id)

    def delete(self, key: str) -> _TxnDelete:
        return _TxnDelete(key=key)


@dataclass
class EtcdLease:
    id: int
    ttl: int


class _WatchThread(threading.Thread):
    def __init__(self, host: str, port: int, key_b64: str, range_end_b64: str, callback: Callable):
        super().__init__(daemon=True)
        self._url = f"http://{host}:{port}/v3/watch"
        self._payload = json.dumps({
            "create_request": {
                "key": key_b64,
                "range_end": range_end_b64,
            }
        }).encode("utf-8")
        self._cb = callback
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                req = urllib.request.Request(
                    self._url,
                    data=self._payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                # No timeout for long streaming connection
                with urllib.request.urlopen(req, timeout=None) as resp:
                    for line in resp:
                        if self._stop_evt.is_set():
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line.decode("utf-8"))
                            # The 'result' field contains 'events' if changed
                            res = msg.get("result", {})
                            if "events" in res:
                                self._cb(res["events"])
                        except Exception:
                            # Ignore malformed chunks
                            pass
            except Exception:
                # Connection lost or failed, backoff and retry
                if self._stop_evt.is_set():
                    break
                time.sleep(1.0)


class EtcdV3HttpClient:
    """Minimal etcd v3 client via HTTP/JSON gateway.

    This avoids protobuf/gRPC client dependencies, and works with etcd 3.x that exposes
    the grpc-gateway endpoints (e.g. /v3/kv/range, /v3/kv/txn, /v3/lease/grant).
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 2379, timeout_s: float = 3.0):
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)

        self.transactions = _TxnOps()

        self._watch_lock = threading.Lock()
        self._watch_counter = 0
        self._active_watches: Dict[int, _WatchThread] = {}

    def _url(self, path: str) -> str:
        p = "/" + (path or "").lstrip("/")
        return f"http://{self._host}:{self._port}{p}"

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except (urllib.error.URLError, socket.timeout) as e:
            raise ConnectionError(f"etcd http request failed: {e}") from e
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as e:
            raise RuntimeError(f"etcd http response is not json: {raw[:200]!r}") from e

    def lease(self, ttl: int) -> EtcdLease:
        out = self._post_json("/v3/lease/grant", {"TTL": int(ttl)})
        lease_id = int(out.get("ID", 0) or 0)
        if lease_id == 0:
            raise RuntimeError(f"failed to grant lease: {out}")
        return EtcdLease(id=lease_id, ttl=int(ttl))

    def lease_keepalive_once(self, lease: EtcdLease | int, *, timeout_s: Optional[float] = None) -> EtcdLease:
        """Best-effort lease keepalive via grpc-gateway streaming endpoint.

        etcd exposes LeaseKeepAlive as a streaming RPC; the HTTP gateway returns a
        stream of JSON objects. To keep this client dependency-free, we do a
        one-shot keepalive: open a request with a single keepalive message,
        read one response frame, then close.

        Returns an updated EtcdLease with refreshed TTL on success.
        """

        lease_id = int(getattr(lease, "id", lease) or 0)
        if lease_id <= 0:
            raise ValueError(f"invalid lease id: {lease_id}")

        data = json.dumps({"ID": str(int(lease_id))}).encode("utf-8")
        req = urllib.request.Request(
            self._url("/v3/lease/keepalive"),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        effective_timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                # Read a single streaming frame.
                raw_line = resp.readline()
        except (urllib.error.URLError, socket.timeout) as e:
            raise ConnectionError(f"etcd lease keepalive failed: {e}") from e

        if not raw_line:
            raise RuntimeError("etcd lease keepalive returned empty response")

        try:
            msg = json.loads(raw_line.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"etcd lease keepalive response is not json: {raw_line[:200]!r}") from e

        # grpc-gateway stream objects typically look like {"result": {...}, "error": {...}}
        res = msg.get("result", msg)
        try:
            ttl = int(res.get("TTL", 0) or 0)
        except Exception:
            ttl = 0
        if ttl <= 0:
            raise RuntimeError(f"etcd lease keepalive returned invalid TTL: {msg}")
        return EtcdLease(id=int(lease_id), ttl=int(ttl))

    def _range(
        self,
        *,
        key: bytes,
        range_end: Optional[bytes] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Tuple[Optional[bytes], EtcdMeta]]:
        payload: dict = {"key": _b64e(bytes(key))}
        if range_end is not None:
            payload["range_end"] = _b64e(bytes(range_end))
        if limit is not None:
            payload["limit"] = int(limit)

        out = self._post_json("/v3/kv/range", payload)
        for kv in out.get("kvs") or []:
            try:
                meta = EtcdMeta(
                    key=_b64d(kv.get("key", "")),
                    create_revision=int(kv.get("create_revision", 0) or 0),
                    mod_revision=int(kv.get("mod_revision", 0) or 0),
                    version=int(kv.get("version", 0) or 0),
                )
                v = _b64d(kv.get("value", "")) if kv.get("value") is not None else None
            except Exception:
                continue
            yield v, meta

    def get(self, key: str) -> Tuple[Optional[bytes], Optional[EtcdMeta]]:
        k = key.encode("utf-8")
        for v, meta in self._range(key=k, limit=1):
            return v, meta
        return None, None

    def get_prefix(self, prefix: str) -> Iterator[Tuple[Optional[bytes], EtcdMeta]]:
        p = prefix.encode("utf-8")
        range_end = _prefix_range_end(p)
        yield from self._range(key=p, range_end=range_end)

    def put(self, key: str, value: str, *, lease=None) -> None:
        lease_id = 0
        if lease is not None:
            lease_id = int(getattr(lease, "id", lease) or 0)
        payload = {
            "key": _b64e(key.encode("utf-8")),
            "value": _b64e(str(value).encode("utf-8")),
        }
        if lease_id:
            payload["lease"] = lease_id
        _ = self._post_json("/v3/kv/put", payload)

    def delete(self, key: str) -> None:
        payload = {"key": _b64e(key.encode("utf-8"))}
        _ = self._post_json("/v3/kv/deleterange", payload)

    def transaction(self, *, compare: Iterable[_TxnCompare], success: Iterable[object], failure: Iterable[object]):
        def _cmp(c: _TxnCompare) -> dict:
            d: dict = {"target": c.target, "key": _b64e(c.key.encode("utf-8")), "result": c.result}
            if c.version is not None:
                d["version"] = int(c.version)
            if c.value is not None:
                d["value"] = _b64e(bytes(c.value))
            return d

        def _req(op: object) -> dict:
            if isinstance(op, _TxnPut):
                r = {
                    "request_put": {
                        "key": _b64e(op.key.encode("utf-8")),
                        "value": _b64e(op.value.encode("utf-8")),
                    }
                }
                if op.lease_id:
                    r["request_put"]["lease"] = int(op.lease_id)
                return r
            if isinstance(op, _TxnDelete):
                return {"request_delete_range": {"key": _b64e(op.key.encode("utf-8"))}}
            raise TypeError(f"unsupported txn op: {type(op)!r}")

        payload = {
            "compare": [_cmp(c) for c in compare],
            "success": [_req(o) for o in success],
            "failure": [_req(o) for o in failure],
        }
        out = self._post_json("/v3/kv/txn", payload)
        return bool(out.get("succeeded", False)), out.get("responses") or []

    def add_watch_prefix_callback(self, prefix: str, callback: Callable) -> int:
        p = prefix.encode("utf-8")
        range_end = _prefix_range_end(p)
        key_b64 = _b64e(p)
        end_b64 = _b64e(range_end)

        t = _WatchThread(self._host, self._port, key_b64, end_b64, callback)
        t.start()

        with self._watch_lock:
            self._watch_counter += 1
            wid = self._watch_counter
            self._active_watches[wid] = t
        return wid

    def cancel_watch(self, watch_id: int) -> None:
        with self._watch_lock:
            t = self._active_watches.pop(watch_id, None)
        if t:
            t.stop()
