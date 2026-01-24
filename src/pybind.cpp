#include "config.h"
#include "core/cache_task.h"
#include "core/error.h"
#include "service/local_cache_service.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(MODULE_NAME, m) {
  using cache::service::LocalCacheService;
  using cache::storage::LocalStorageEngine;
  using cache::task::CacheTask;
  using cache::task::State;

  py::enum_<State>(m, "State")
      .value("Initial", State::Initial)
      .value("Working", State::Working)
      .value("Finished", State::Finished)
      .value("Aborted", State::Aborted)
      .export_values();

  py::class_<LocalStorageEngine::HashInfo, std::shared_ptr<LocalStorageEngine::HashInfo>>(m, "LocalCacheHashInfo");

  py::class_<CacheTask, std::shared_ptr<CacheTask>>(m, "Task")
      .def("ready", &CacheTask::ready, "Check if task is ready")
      .def("data_safe", &CacheTask::data_safe, "Check if data is safe (for write mode: data copied from KV cache)")
      .def("state", &CacheTask::state, "Get task block states")
      .def("get_page_already_list", &CacheTask::get_page_already_list, "Get list of page indices already on disk");

  py::class_<LocalCacheService>(m, "LocalCacheService")
      .def(py::init<const std::string &, std::size_t, std::size_t, const torch::Tensor &, std::size_t,
                    const std::string &, bool, const std::string &>(),
           py::arg("file"), py::arg("storage_size"), py::arg("num_of_shard"), py::arg("kvcache"),
           py::arg("num_workers"), py::arg("index_endpoint") = std::string(), py::arg("bandwidth_log") = true,
           py::arg("index_prefix") = std::string())
      .def("run", &LocalCacheService::run)
      .def("query", &LocalCacheService::query)
      .def("update_shard_assignments", &LocalCacheService::update_shard_assignments, py::arg("shard_ids"),
           py::arg("epochs"), py::arg("draining"))
      .def("recover_shard_to_redis", &LocalCacheService::recover_shard_to_redis, py::arg("shard_id"))
      .def("recover_shard_to_redis_smart", &LocalCacheService::recover_shard_to_redis_smart, py::arg("shard_id"))
      .def("shard_inflight", &LocalCacheService::shard_inflight, py::arg("shard_id"))
      .def("shard_eviction_count", &LocalCacheService::shard_eviction_count, py::arg("shard_id"))
      .def("eviction_count", &LocalCacheService::eviction_count)
      .def("eviction_observed", &LocalCacheService::eviction_observed)
      .def("abort", &LocalCacheService::abort_task)
      .def("create", &LocalCacheService::create)
      .def("active_create_count", &LocalCacheService::active_create_count, py::arg("mode"))
      .def("block_size", &LocalCacheService::block_size)
      .def("page_size", &LocalCacheService::page_size);
}