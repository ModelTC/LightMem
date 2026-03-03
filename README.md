# LightMem

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

LightMem is a high-performance KV cache management library designed for large language model (LLM) inference systems. It provides efficient disk-based caching solutions for key-value pairs, enabling memory-efficient long-context processing with minimal performance overhead.

## Project Overview

LightMem serves as a storage optimization layer for LLM inference frameworks, offering:
- **Disk-Based KV Cache**: Persistent storage of key-value cache with efficient read/write operations
- **Asynchronous I/O**: Non-blocking cache operations using multi-threaded task queues
- **Memory Efficiency**: Reduced GPU/CPU memory footprint by offloading KV cache to disk
- **Scalability**: Support for large-scale inference workloads with configurable storage sharding
- **Multi-Node Support**: Distributed shard ownership and cross-node deduplication via etcd + Redis

## Key Features

### Core Modules
| Module              | Description                                                                                        |
|---------------------|----------------------------------------------------------------------------------------------------|
| **Storage**         | Pluggable storage engine interface with local file system implementation                           |
| **Service**         | Cache service layer managing read/write operations with task scheduling                            |
| **Task Queue**      | Asynchronous task processing system with configurable worker threads                               |
| **Core**            | Cache block management and task state tracking for reliable operations                             |
| **Index (Redis)**   | Global hash index for cross-node deduplication and fast cache lookup                               |
| **Coordinator (etcd)** | Distributed shard ownership via Rendezvous (HRW) hashing with lease-based failure detection    |

### Architecture Highlights
- **Block-Level Management**: KV cache divided into fixed-size blocks for efficient I/O
- **Hash-Based Indexing**: Fast cache lookup using content-based hashing
- **Zero-Copy Design**: Direct memory mapping between PyTorch tensors and storage
- **Thread-Safe Operations**: Concurrent read/write support with fine-grained locking
- **HRW Shard Assignment**: Deterministic Rendezvous hashing minimizes shard movement on node join/leave
- **Epoch Fencing**: Write operations are bound to shard ownership epochs to prevent stale writes
- **Delayed Handoff (Drain)**: Ownership transfer waits for in-flight operations to complete before releasing

## Installation

### System Requirements
- Python 3.10 or higher
- CMake 3.25 or higher
- C++17 compatible compiler
- PyTorch (with CPU support)
- Boost C++ Libraries
- pybind11 (automatically installed via pip dependencies)

**Platform Notes:**
- **Linux**: Full support with optimized page cache management via `posix_fadvise`
- **macOS**: Supported, but without `posix_fadvise` optimization (not available on macOS)

### Installation Methods

#### Install system dependencies

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake build-essential libboost-all-dev
```

**On macOS:**
```bash
brew install cmake boost
```

**Using Conda (Cross-platform):**
```bash
conda install -c conda-forge cmake cxx-compiler boost libboost-devel
```

**Install PyTorch:**
```bash
pip install torch
```

#### Install optional dependencies for multi-node mode

**Docker (recommended — one command starts everything):**
- [Install Docker Engine](https://docs.docker.com/engine/install/) and the `docker compose` plugin
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

**Local binaries (alternative to Docker):**

On Ubuntu/Debian:
```bash
sudo apt-get install redis-server etcd
```

On macOS:
```bash
brew install redis etcd
```

On Conda:
```bash
conda install -c conda-forge redis-server etcd
```

#### Using pip (Recommended)
```bash
pip install -v .
```

### Environment Variables

#### `LIGHTMEM_MAX_BLOCK_SIZE_MB`
Controls the maximum size of each cache block in megabytes (MB).

- **Default**: `64` (64MB)
- **Purpose**: Determines the granularity of cache I/O operations. Each cache block is read from or written to disk as a single unit.
- **Usage**:
  ```bash
  export LIGHTMEM_MAX_BLOCK_SIZE_MB=32  # Set to 32MB
  ```
- **Considerations**:
  - **Larger blocks** (e.g., 128): Reduce overhead, better for sequential access, but may increase latency for small operations
  - **Smaller blocks** (e.g., 16): More fine-grained control, better for random access, but higher overhead per operation
  - Must be set before starting the cache service

#### Index Persistence (Optional)

LightMem can persist the hash index to an external Redis backend. When LightMem restarts (even after a process crash), as long as Redis is still running, LightMem can rebuild the in-memory hash index from it and continue using the existing local disk cache files.

It also writes recovery metadata to each shard's `meta` file using **SuperBlock + Journal (truncate mode)**. If Redis is not available (or index data is missing), LightMem falls back to replaying the local journal to rebuild the index.

Enable by passing `index_endpoint` to `PyLocalCacheService`:

```python
# Single-host default ports (Redis: 6379, etcd: 2379)
svc = PyLocalCacheService(
    kvcache_tensor=kvcache_tensor,
    file=file,
    index_endpoint="127.0.0.1",   # host-only: auto-use default ports
)

# Explicit ports
svc = PyLocalCacheService(
    kvcache_tensor=kvcache_tensor,
    file=file,
    index_endpoint="127.0.0.1:6379",
    coord_endpoints="127.0.0.1:2379",
)
```

#### `lightmem_server` (one command)

After installing LightMem, start all dependency services with a single command:

```bash
# Start Redis (index) + etcd (coordinator) via Docker Compose
lightmem_server --mode docker --index-port 6379 --coord-port 2379 --coord-peer-port 2380

# Start Redis (index) + etcd (coordinator) via local platform
lightmem_server --mode local --index-port 6379 --coord-port 2379 --coord-peer-port 2380

# Stop services and remove persistent volumes
lightmem_server --stop --purge-volumes
```

`lightmem_server` prints a ready-to-use Python snippet for `PyLocalCacheService` clients once services are up.

## Quick Start

### Key Concepts

**Data, Hashes, Pages, and Blocks:**
- Each **data element** is hashed using 128-bit cumulative hashing (xxHash3)
- Each **cumulative hash** corresponds to one **page** in the KV cache (one-to-one mapping via `kv_page_indexer`)
- **Cumulative hash**: Each position contains the hash of all data from the start up to that position
- Hashes are automatically grouped into **blocks** for I/O operations
- **Block size** = `LIGHTMEM_MAX_BLOCK_SIZE_MB` (default 64MB)
- **Pages per block** = block_size / page_size
- Example: With 64MB blocks and 16KB pages, each block contains ~4096 pages
- For block operations: The last cumulative hash of each block represents the entire block

### Basic Usage

```python
import torch
from light_mem import PyLocalCacheService

# Create a CPU-based KV cache tensor
# Shape: [num_pages, page_size] - must be 2D uint8 tensor
kv_cache = torch.zeros((1000, 40 * 8192), dtype=torch.float16).view(dtype=torch.uint8)

# Initialize cache service
cache_service = PyLocalCacheService(
    kvcache_tensor=kv_cache,     # KV cache tensor (2D uint8)
    file="./cache_storage",      # Storage directory
    storage_size=10 * 1024**3,   # 10GB storage limit
    num_shard=4,                 # Number of storage shards
    num_worker=8                 # Number of worker threads
)

# Service starts automatically after initialization

# Compute cumulative hashes from data
hash_128s = [hash_1, hash_2, hash_3, hash_4]  # list of 128-bit integers (cumulative hashes)

# Query if caches exist (returns list of booleans)
exists_list = cache_service.query(hash_128s)

# Create write/read tasks
# Note: hash_128s and kv_page_indexer must have the same length (one-to-one mapping)
task = cache_service.create(
    hash_128s=hash_128s,          # List of 128-bit cumulative hash integers
    kv_page_indexer=torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # Page indices (same length as hash_128s)
    mode="w"                      # "w" for write, "r" for read
)

# Check task status
if task.ready():
    print("Task completed!")
```

### Task Management

```python
# Check task state
states = task.state()  # Returns PyState enum list for each block

# Abort a running task
cache_service.abort(task)

# Get pages already cached on disk (write mode only)
cached_pages = task.page_already_list  # Property, not method

# Check if data is safe to modify (for write tasks)
if task.data_safe():
    # Safe to modify source tensor
    pass
```

## Architecture

LightMem has a layered architecture with C++ core and Python bindings:

### Python Layer
- **PyLocalCacheService**: Main Python interface for cache operations
- **PyTask**: Python wrapper for task management
- **PyState**: Enum for task state tracking (Initial, Working, Finished, Aborted)
- **EtcdShardCoordinator**: Background thread managing distributed shard ownership via etcd

### C++ Core (Internal)

#### Storage Layer
- **StorageEngine**: Abstract interface for pluggable storage backends
- **LocalStorageEngine**: File-based storage implementation with sharding support

#### Service Layer
- **CacheService**: Base class defining cache service interface
- **LocalCacheService**: Concrete implementation managing local disk cache

#### Task Processing
- **CacheTask**: Represents a complete read/write operation
- **CacheBlock**: Individual block within a task, processed independently
- **TaskQueue**: Thread pool managing asynchronous task execution

## Multi-Node Distributed Cache

Multiple nodes share the same storage infrastructure: each node independently manages local shard files, coordinates ownership through **etcd**, and deduplicates via a **Redis** global hash index.

- **Shard ownership**: Assigned via Rendezvous (HRW) hashing — deterministic, minimal movement on join/leave, single owner per shard at any time
- **Ownership lifecycle**: `FREE → CLAIMED → DRAINING → FREE`. Owner keys are lease-bound; a crashed node's shards are reassigned automatically after `coord_ttl` seconds
- **Epoch fencing**: Each claim generates a monotonic epoch stored in the shard WAL/superblock, preventing stale writes from a previous owner
- **Cross-node dedup**: On write, each node checks `lightmem:global:index` in Redis; if another node already wrote the same hash, the write is skipped

### Setup

```bash
# 1. Start Redis + etcd (on any accessible host)
lightmem_server --mode docker --index-port 6379 --coord-port 2379 --coord-peer-port 2380
```

```python
# 2. Start each node (num_shard must be identical across all nodes)
svc = PyLocalCacheService(
    kvcache_tensor=kv_cache,
    file="./cache_storage",
    storage_size=100 * 1024**3,
    num_shard=128,
    index_endpoint="192.168.1.10",        # Redis host (default port 6379)
    index_prefix="light_mem",
)

# 3. Graceful shutdown
svc.close()
```

> Multiple processes on the **same machine** must use different `coord_node_id` values.

## Configuration

### Block Size Configuration
The block size determines the granularity of cache operations:
```bash
# Default: 64MB per block
# Override via environment variable (value in MB)
export LIGHTMEM_MAX_BLOCK_SIZE_MB=128  # Set to 128MB
```

### Storage Sharding
Distribute cache files across multiple shards for better I/O parallelism:
```python
num_shard=8  # Creates 8 separate storage files
```

## Performance Considerations

- **Worker Threads**: More workers improve I/O parallelism but increase CPU overhead
- **Block Size**: Larger blocks reduce overhead but may increase latency for small operations
- **Storage Sharding**: More shards improve concurrent access but increase file descriptor usage
- **Memory Alignment**: KV cache tensors must be contiguous for optimal performance

## API Reference

### PyLocalCacheService

```python
PyLocalCacheService(
    kvcache_tensor: torch.Tensor,  # 2D uint8 tensor [num_pages, page_size], CPU, contiguous
    file: str,                     # storage directory path
    storage_size: int = 32 * 1024**3,
    num_shard: int = 32,
    num_worker: int = 16,
)
```

- `query(hash_128s) -> List[bool]`: check if blocks exist on disk (one bool per block)
- `create(hash_128s, kv_page_indexer, mode, start_pos=0) -> PyTask`: submit async read (`"r"`) or write (`"w"`) task; `kv_page_indexer` is an `int32` tensor same length as `hash_128s`
- `abort(task)`: cancel a task

### PyTask

- `ready() -> bool`: all blocks done (finished or aborted)
- `data_safe() -> bool`: source tensor pages can be safely reused (write: data buffered; read: same as `ready()`)
- `state() -> List[PyState]`: per-block state — `Initial / Working / Finished / Aborted`

## Contributing

Contributions are welcome! Please ensure:
- Code follows C++17 and Python 3.10+ standards
- All tests pass before submitting PRs
- Documentation is updated for new features

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

LightMem is developed as part of the ModelTC ecosystem for efficient LLM inference.