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

## Key Features

### Core Modules
| Module          | Description                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------|
| **Storage**     | Pluggable storage engine interface with local file system implementation                        |
| **Service**     | Cache service layer managing read/write operations with task scheduling                         |
| **Task Queue**  | Asynchronous task processing system with configurable worker threads                            |
| **Core**        | Cache block management and task state tracking for reliable operations                          |

### Architecture Highlights
- **Block-Level Management**: KV cache divided into fixed-size blocks for efficient I/O
- **Hash-Based Indexing**: Fast cache lookup using content-based hashing
- **Zero-Copy Design**: Direct memory mapping between PyTorch tensors and storage
- **Thread-Safe Operations**: Concurrent read/write support with fine-grained locking

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

#### Using pip (Recommended)
```bash
pip install -v .
```

#### Build and install from source
```bash
# Build wheel package
python -m build --wheel

# Install the built wheel
pip install dist/*.whl
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

## Quick Start

### Key Concepts

**Tokens, Pages, and Blocks:**
- Each **token** corresponds to one **page** in the KV cache (one-to-one mapping via `kv_page_indexer`)
- Tokens are automatically grouped into **blocks** for I/O operations
- **Block size** = `LIGHTMEM_MAX_BLOCK_SIZE_MB` (default 64MB)
- **Pages per block** = block_size / page_size
- Example: With 64MB blocks and 16KB pages, each block contains ~4096 pages (tokens)
- Hash values are computed per block, not per token

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

# Query if caches exist (returns list of booleans)
# Pass token list - hashes are computed automatically
tokens = [100, 200, 300, 400]  # Example token IDs
exists_list = cache_service.query(tokens)

# Create write/read tasks
# Note: tokens and kv_page_indexer must have the same length (one-to-one mapping)
task = cache_service.create(
    tokens=tokens,                # List of token IDs
    kv_page_indexer=torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # Page indices (same length as tokens)
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

#### Constructor
```python
PyLocalCacheService(
    kvcache_tensor: torch.Tensor,
    file: str,
    storage_size: int = 32 * 1024**3,  # Default: 32GB
    num_shard: int = 32,
    num_worker: int = 16
)
```
**Parameters:**
- `kvcache_tensor`: 2D uint8 tensor with shape `[num_pages, page_size]`, must be CPU and contiguous
- `file`: Path to storage directory/file
- `storage_size`: Total storage size in bytes (distributed across shards)
- `num_shard`: Number of storage file shards
- `num_worker`: Number of worker threads

#### Methods
- `hash(tokens: List[int]) -> List[str]`: Compute hash values for token list
  - Tokens are automatically grouped into blocks (n tokens per block, where n = block_size / page_size)
  - Returns one hash string per block
  - Example: 100 tokens with n=4096 → returns 1 hash (since 100 < 4096)
  - Example: 5000 tokens with n=4096 → returns 2 hashes (4096 + 904 tokens)
- `query(tokens: List[int]) -> List[bool]`: Check if caches exist for tokens, returns list of booleans
  - Returns one boolean per block (not per token)
  - Use `hash(tokens)` to see how many blocks are queried
- `create(tokens: List[int], kv_page_indexer: torch.Tensor, mode: str, start_pos: int = 0) -> PyTask`: Create cache task
  - `tokens`: List of token IDs (hashes computed automatically)
  - `kv_page_indexer`: Int32 tensor containing page indices, **must have the same length as tokens** (one-to-one mapping)
  - `mode`: `"w"` for write, `"r"` for read
  - `start_pos`: Optional starting position in token list (default: 0)
- `abort(task: PyTask)`: Cancel a running task
- `active_threads(mode: str) -> int`: Get count of active read/write tasks (`"w"` or `"r"`)

### PyTask

#### Methods
- `ready() -> bool`: Check if all blocks are finished
- `data_safe() -> bool`: Check if source data can be safely modified (write: data copied; read: equivalent to ready())
- `state() -> List[PyState]`: Get PyState enum for each block
  - `PyState.Initial` (0): Task just created
  - `PyState.Working` (1): Task in progress
  - `PyState.Finished` (2): Task completed successfully
  - `PyState.Aborted` (3): Task aborted (possibly due to error)

#### Properties
- `page_already_list -> List[int]`: Get list of page indices already on disk (write mode: pages found in cache via hash query)

## Contributing

Contributions are welcome! Please ensure:
- Code follows C++17 and Python 3.10+ standards
- All tests pass before submitting PRs
- Documentation is updated for new features

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

LightMem is developed as part of the ModelTC ecosystem for efficient LLM inference.