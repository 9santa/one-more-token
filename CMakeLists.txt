cmake_minimum_required(VERSION 3.16)

project(one_more_token LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(one_more_token INTERFACE)

# target_link_libraries(one_more_token INTERFACE CUDA::cudart)

target_include_directories(one_more_token INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(one_more_token_cuda STATIC
    src/cuda/vector_add.cu
    # src/model/cuda_mock_model.cu
)

target_include_directories(one_more_token_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(one_more_token_cuda PUBLIC
    one_more_token
)

set_target_properties(one_more_token_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(run_cuda_vector_add
    examples/run_cuda_vector_add.cpp
)
target_link_libraries(run_cuda_vector_add PRIVATE one_more_token_cuda)

# add_executable(run_single_request
#     examples/run_single_request.cpp
# )
# target_link_libraries(run_single_request PRIVATE one_more_token)
#
#
# add_executable(run_multi_request
#     examples/run_multi_request.cpp
# )
# target_link_libraries(run_multi_request PRIVATE one_more_token)
#
#
# add_executable(run_scheduled_engine
#     examples/run_scheduled_engine.cpp
# )
# target_link_libraries(run_scheduled_engine PRIVATE one_more_token)


add_executable(run_prefill_decode_engine
    examples/run_prefill_decode_engine.cpp
)
target_link_libraries(run_prefill_decode_engine PRIVATE one_more_token)
