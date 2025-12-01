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

### Basic Usage

```python
import torch
from light_mem import LocalCacheService

# Create a CPU-based KV cache tensor
# Shape: [num_pages, num_layers, page_size]
kv_cache = torch.zeros((1000, 40 * 8192), dtype=torch.float16).view(dtype=torch.uint8)

# Initialize cache service
cache_service = LocalCacheService(
    file="./cache_storage",      # Storage directory
    storage_size=10 * 1024**3,   # 10GB storage limit
    num_of_shard=4,              # Number of storage shards
    kvcache=kv_cache,            # KV cache tensor
    num_workers=8                # Number of worker threads
)

# Start the service
cache_service.run()

# Query if a cache exists
exists = cache_service.query("hash_key")

# Create write/read tasks
task = cache_service.create(
    mode="write",                 # or "read"
    hash_values=["hash1", "hash2"],
    page_indices=[0, 1]
)

# Check task status
if task.ready():
    print("Task completed!")
```

### Task Management

```python
# Check task state
states = task.state()  # Returns state for each block

# Abort a running task
cache_service.abort(task)

# Get pages already cached on disk
cached_pages = task.get_page_already_list()

# Check if data is safe to modify (for write tasks)
if task.data_safe():
    # Safe to modify source tensor
    pass
```

## Architecture

### Storage Layer
- **StorageEngine**: Abstract interface for pluggable storage backends
- **LocalStorageEngine**: File-based storage implementation with sharding support

### Service Layer
- **CacheService**: Base class defining cache service interface
- **LocalCacheService**: Concrete implementation managing local disk cache

### Task Processing
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
num_of_shard=8  # Creates 8 separate storage files
```

## Performance Considerations

- **Worker Threads**: More workers improve I/O parallelism but increase CPU overhead
- **Block Size**: Larger blocks reduce overhead but may increase latency for small operations
- **Storage Sharding**: More shards improve concurrent access but increase file descriptor usage
- **Memory Alignment**: KV cache tensors must be contiguous for optimal performance

## API Reference

### LocalCacheService

#### Constructor
```python
LocalCacheService(file: str, storage_size: int, num_of_shard: int, 
                  kvcache: torch.Tensor, num_workers: int)
```

#### Methods
- `run()`: Start the cache service worker threads
- `query(hash: str) -> bool`: Check if cache exists
- `create(mode: str, hash_values: List[str], page_indices: List[int]) -> Task`: Create cache task
- `abort(task: Task)`: Cancel a running task
- `block_size() -> int`: Get block size in bytes
- `page_size() -> int`: Get page size in bytes

### Task

#### Methods
- `ready() -> bool`: Check if all blocks are finished
- `data_safe() -> bool`: Check if source data can be safely modified
- `state() -> List[State]`: Get state of each block
- `get_page_already_list() -> List[int]`: Get list of cached page indices

## Contributing

Contributions are welcome! Please ensure:
- Code follows C++17 and Python 3.10+ standards
- All tests pass before submitting PRs
- Documentation is updated for new features

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

LightMem is developed as part of the ModelTC ecosystem for efficient LLM inference.