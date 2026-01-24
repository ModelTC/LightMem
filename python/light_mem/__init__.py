from enum import Enum
from typing import List, Optional
import torch

from . import light_mem
from .etcd_coordinator import maybe_start_etcd_coordinator


class PyState(Enum):
    """ PyState 枚举类型用来描述任务块 (TaskBlock) 的执行状态
    用户需要在任务执行结束后，手动查看每一个任务块的执行状态，
    只有状态为 Finished 的任务块表示任务成功执行，对应的数据有效
    """
    # 任务刚刚完成创建
    Initial = 0
    # 任务正在进行中
    Working = 1
    # 任务成功执行完成
    Finished = 2
    # 任务被终止 (可能因为执行出错)
    Aborted = 3


class PyTask:
    """ PyTask 对象用于描述一个异步存取任务
    你不应该调用这个类型的构造函数，所有 PyTask 都应当由 PyLocalCacheService.create 函数创建
    一旦完成创建，任务便立即会被提交到 PyLocalCacheService 的任务队列并异步地执行

    每一个 PyTask 在底层都会按照 BlockSize 的约定，分解成一系列的 Task Block
    PyLocalCacheService 将以 Block 为单位完成任务的读写

    用户需要在合适的时间调用 ready() 函数查看 PyTask 的执行情况
    如果 ready() 函数返回 True，则说明任务已经执行结束
    此时调用 state() 函数将返回所有任务块的执行情况
    """
    def __init__(self, _c):
        self._c = _c
        self._state_convert = {
            light_mem.State.Initial: PyState.Initial,
            light_mem.State.Working: PyState.Working,
            light_mem.State.Finished: PyState.Finished,
            light_mem.State.Aborted: PyState.Aborted
        }

    def ready(self) -> bool:
        """ ready 函数用来询问任务是否完成执行 """
        return self._c.ready()

    def data_safe(self) -> bool:
        """ data_safe 函数用来询问数据是否已经安全可以释放页面
        对于写模式：数据已从 KV cache 拷贝完成，页面可以安全释放（无需等待磁盘写入完成）
        对于读模式：等同于 ready()
        """
        return self._c.data_safe()

    def state(self) -> List[PyState]:
        """ state 函数用来询问所有任务块的执行情况 """
        return [self._state_convert[s] for s in self._c.state()]

    @property
    def page_already_list(self) -> List[int]:
        """ 返回已经完成落盘的page索引列表（写模式下，hash查询发现在disk cache里已经有的内容）"""
        return self._c.get_page_already_list()


class PyLocalCacheService:
    """ 基于本地存储的异步数据存取服务 """
    @staticmethod
    def _normalize_host_only(value: str) -> str:
        v = (value or "").strip()
        if not v:
            return ""
        if v.startswith("http://"):
            v = v[len("http://"):]
        elif v.startswith("https://"):
            v = v[len("https://"):]
        # If someone accidentally passes a URL, keep only the authority.
        if "/" in v:
            v = v.split("/", 1)[0]

        # Allow bracketed IPv6 (e.g. "[::1]").
        if ":" in v and not v.startswith("["):
            raise ValueError("host must be host/ip without port")
        return v

    def __init__(
    self,
    kvcache_tensor: torch.Tensor,
    file: str,
    storage_size: int = 32*1024*1024*1024,
    num_shard: int = 32,
    num_worker: int = 16,
    *,
    index_endpoint: str = "",
    index_prefix: str = "lightmem",
    bandwidth_log: bool = True,
    coord_endpoints: str = "",
    coord_node_id: Optional[str] = None,
    coord_ttl: int = 10,
    coord_reconcile_sec: float = 10.0,
    ):
        """ 使用 PyLocalCacheService 来创建异步数据存取引擎 (基于本地磁盘)
    PyLocalCacheService 会直接从 kv cache 中读取数据，并异步地向 kv cache 中写入数据
    kv cache tensor 必须是二维张量，形如 [num of page, page size], 类型为 uint8
    kv cache tensor 必须是连续内存
    kv cache tensor 必须是 CPU 张量

        Args:
            file (str): 本地文件位置
            num_shard(int): 本地文件分片数量
            storage_size (int): 本地文件大小 (本地文件是分片存储的，这里表示总大小)
            kvcache_tensor (torch.Tensor): kvcache tensor，维度顺序为 [page, stride]
            num_worker (int): 工作线程数量
            index_endpoint (str): Redis 索引服务地址。
                - 若仅提供主机名/IP (如 "127.0.0.1")，默认使用 6379 端口，并尝试推断 coord_endpoints 为该主机:2379。
                - 若提供主机名:端口 (如 "127.0.0.1:6379")，则严格作为 Redis 地址，此时若需 ETCD 必须显式指定 coord_endpoints。
                - 为空则禁用索引和分布式功能。
            index_prefix (str): 索引和协调器使用的 Key 前缀。
            bandwidth_log (bool): 是否打印带宽统计日志。
            coord_endpoints (str): ETCD 协调服务地址 (用于分片所有权和分布式锁)。若为空且 index_endpoint 为纯主机名，会自动推断。
            coord_node_id (str, optional): 分布式协调中的当前节点 ID。
                - 若为 None，默认使用 socket.gethostname()。
                - 注意：如果在同一台机器上运行多个进程，必须手动指定不同的 ID 以避免冲突。
            coord_ttl (int): ETCD 租约 TTL (秒)，决定节点故障判定时间。默认 10 秒。
            coord_reconcile_sec (float): 协调器兜底轮询周期 (秒)。默认约为 TTL(秒)。
                - watch 可能因网络抖动/压缩等原因短暂不可用，此轮询用于保证最终收敛。
                - 若分片数量较大 (如 >500)，可适当调大以减轻 ETCD 压力。
        """
        if kvcache_tensor.dim() != 2:
            raise ValueError("kvcache_tensor 必须是二维张量，形如 [num of page, page size]")
        if kvcache_tensor.is_cuda:
            raise ValueError("kvcache_tensor 必须是 CPU 张量，GPU 流程已移除")

        num_pages_total = kvcache_tensor.shape[0]

        index_endpoint_str = (index_endpoint or "").strip()
        index_prefix_str = (index_prefix or "").strip()
        coord_endpoints_str = (coord_endpoints or "").strip()

        # Parameter minimization: index_endpoint can be either:
        # - ""                 (disable index)
        # - "host"             (treat as deps host; use default ports 6379/2379)
        # - "host:port"        (explicit index endpoint)
        host_only = ""
        if index_endpoint_str:
            if index_endpoint_str.startswith("["):
                # Bracketed IPv6 forms:
                # - "[::1]"          (host-only)
                # - "[::1]:6379"     (explicit)
                if "]:" in index_endpoint_str:
                    host_only = ""
                elif index_endpoint_str.endswith("]"):
                    host_only = self._normalize_host_only(index_endpoint_str)
                else:
                    raise ValueError("index_endpoint IPv6 must be '[addr]' or '[addr]:port'")
            elif ":" not in index_endpoint_str:
                host_only = self._normalize_host_only(index_endpoint_str)

        if host_only:
            index_endpoint_str = f"{host_only}:6379"
            if not coord_endpoints_str:
                coord_endpoints_str = f"{host_only}:2379"

        self._c = light_mem.LocalCacheService(
            file=file,
            storage_size=storage_size,
            num_of_shard=num_shard,
            kvcache=kvcache_tensor,
            num_workers=num_worker,
            index_endpoint=index_endpoint_str,
            bandwidth_log=bool(bandwidth_log),
            index_prefix=index_prefix_str,
        )
        self._c.run()

        # Optional: start coordinator-driven shard ownership.
        self._etcd_thread = maybe_start_etcd_coordinator(
            self._c,
            num_shards=num_shard,
            endpoints=coord_endpoints_str,
            index_prefix=index_prefix_str,
            node_id=coord_node_id,
            coord_ttl=int(coord_ttl),
            coord_reconcile_sec=float(coord_reconcile_sec),
        )
        self._num_of_page_total: int = num_pages_total
        self._block_size: int = int(self._c.block_size())
        self._page_size: int = int(self._c.page_size())
        self._n: int = self._block_size // self._page_size

    def close(self) -> None:
        """Best-effort shutdown for background coordinator thread.

        Note: the underlying C++ LocalCacheService currently has no explicit stop() binding.
        This method focuses on stopping the etcd lease keepalive thread so that keys can
        expire and tests can deterministically emulate node leave.
        """
        t = getattr(self, "_etcd_thread", None)
        if t is None:
            return
        try:
            t.stop()
        except Exception:
            pass
        try:
            t.join(timeout=2.0)
        except Exception:
            pass
        self._etcd_thread = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _hash(self, hash_128s: List[int]) -> List[str]:
        """将 128 位哈希整数数组按照长度 n 进行划分，并生成块哈希。

        由于输入是累计哈希（每个哈希已包含之前所有数据的信息），
        直接使用每个块的最后一个哈希值作为该块的标识即可。

        :param hash_128s: 128 位累计哈希整数列表。
        :return: 每个分块的哈希值列表（32位十六进制字符串）。
        """
        if not hash_128s:
            return []

        n = self._n
        num_full = len(hash_128s) // n

        # 处理完整块 - 取每个块的最后一个哈希值
        result = [format(hash_128s[(i+1)*n - 1], '032x')
                  for i in range(num_full)]

        # 处理剩余块 - 取剩余部分的最后一个哈希值
        if len(hash_128s) % n:
            result.append(format(hash_128s[-1], '032x'))

        return result

    def create(
        self, hash_128s: List[int], kv_page_indexer: torch.Tensor, mode: str, start_pos: int = 0
    ) -> PyTask:
        """ 创建并立即提交一个异步存取任务

        Args:
            hash_128s (List[int]): 用户传入的 128 位哈希整数列表
                (实际执行的存取请求面向 hash_128s[start_pos: ])
                但用户不能够直接传入截断后的 hash_128s，因为需要 hash_128s 前面的内容计算哈希
            start_pos (int): 任务开始位置
            kv_page_indexer (torch.Tensor):
                一个索引数组，用来表示 hash_128s 对应的 kvcache_tensor 位置，必须是一维的
            mode (str): r: 异步读取 | w: 异步写入

        Returns:
            PyTask: 一个任务对象
        """

        # 我们需要根据 start pos, end pos 来确定究竟多少个 block 需要被提交
        # 并且针对真实提交的任务，确定其对应的 kv_page_indexer
        start_block_idx: int = start_pos // self._n
        hashs = self._hash(hash_128s=hash_128s)
        hashs = hashs[start_block_idx: ]
        _kv_page_indexer = kv_page_indexer[start_block_idx * self._n: ]

        task = self._c.create(hashs, _kv_page_indexer, mode)
        return PyTask(task)

    def query(self, hash_128s: List[int]) -> List[bool]:
        """ 查询给定的哈希是否存在(以hash列表形式一次查询多个) """
        hashs = self._hash(hash_128s=hash_128s)
        return self._c.query(hashs)

    def active_threads(self, mode: str) -> int:
        """统计当前处于执行中的读写任务数量，mode 使用 "r" 或 "w"""
        return int(self._c.active_create_count(mode))

    def eviction_count(self) -> int:
        """返回本节点发生过的真实 LRU 淘汰次数（跨所有 shard 累加）。"""
        return int(self._c.eviction_count())

    def eviction_observed(self) -> bool:
        """是否已经开始触发真实 LRU 淘汰（本节点）。"""
        return bool(self._c.eviction_observed())

    def abort(self, t: PyTask):
        """ 终止一个任务的执行，此函数调用后，
        工作线程不会立即停止工作(因为他们是异步的，可能有正在进行的任务)，
        但保证所有工作线程不再会从缓存中读写内容
        """
        self._c.abort(t._c)

__all__ = [
    "PyState", "PyTask", "PyLocalCacheService",
]
