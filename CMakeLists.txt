cmake_minimum_required(VERSION 3.25)
project(kvcache LANGUAGES CXX)

find_package(Threads REQUIRED)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
