cmake_minimum_required(VERSION 3.16)

project(one_more_token LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(one_more_token INTERFACE)

target_include_directories(one_more_token INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(run_single_request
    examples/run_single_request.cpp
)
target_link_libraries(run_single_request PRIVATE one_more_token)


add_executable(run_multi_request
    examples/run_multi_request.cpp
)
target_link_libraries(run_multi_request PRIVATE one_more_token)
