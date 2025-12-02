set(name ${PROJECT_NAME})

# dependencies
find_package(Python 3 REQUIRED COMPONENTS Development Interpreter)

# Get site-packages directory from Python
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE Python_SITELIB
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Derive Torch CMake path from site-packages
set(_torch_cmake_path "${Python_SITELIB}/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_path}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Boost REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# sources list
set(source
    pybind.cpp)

# library
Python_add_library(${name} MODULE ${source} WITH_SOABI)
target_include_directories(${name} PRIVATE .)
target_compile_definitions(${name} PRIVATE MODULE_NAME=${name})
target_compile_features(${name} PRIVATE cxx_std_17)
target_link_libraries(${name} PRIVATE Python::Python Boost::boost ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(${name} PROPERTIES
  CXX_EXTENSIONS OFF
  PREFIX "")

if(DEFINED PY_BUILD_CMAKE_IMPORT_NAME)
  install(TARGETS ${name} DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})
endif()
