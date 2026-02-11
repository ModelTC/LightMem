cmake_minimum_required(VERSION 3.25)
project(light_mem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LIGHTMEM_ENABLE_ASAN "Enable AddressSanitizer (ASan)" OFF)

find_package(Threads REQUIRED)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
