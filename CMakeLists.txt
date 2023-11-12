set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

cmake_minimum_required(VERSION 2.8.3)
project(twinlitenet_onnx)

# Options
option(ENABLE_CUDA "Enable CUDA" OFF)

if(ENABLE_CUDA)
    message(STATUS "CUDA is enabled")
    add_definitions(-DENABLE_CUDA)
endif()

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

set(HEADER_FILES
    include/twinlitenet_onnxruntime.hpp)
set(SOURCE_FILES
    src/main.cpp
    src/twinlitenet_onnxruntime.cpp)

# general Includes
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# cuda includes
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )
endif()

add_executable(main ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(main
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime)

# cuda libs
if(ENABLE_CUDA)
    target_link_libraries(main
        ${CUDA_LIBRARIES})
endif()


