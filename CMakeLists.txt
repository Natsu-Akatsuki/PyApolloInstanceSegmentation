cmake_minimum_required(VERSION 3.14)
project(lidar_apollo_instance_segmentation)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)

include(cmake/TensorRT.cmake)


find_package(PCL REQUIRED)
find_package(FMT REQUIRED)
find_package(yaml-cpp REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")


# >>> pybind11 >>>
# get python version
execute_process(
  COMMAND python3 --version
  COMMAND cut -c 8-
  OUTPUT_VARIABLE PYTHON_VERSION)
string(STRIP ${PYTHON_VERSION} PYTHON_VERSION)
set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})

message(STATUS "Python version is ${PYBIND11_PYTHON_VERSION}")
find_package(pybind11)
# >>> pybind11 >>>

include_directories(
  include
  lib/include
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

add_library(tensorrt_apollo_cnn_lib SHARED
  lib/src/TrtNet.cpp
  )

target_link_libraries(tensorrt_apollo_cnn_lib
  ${NVINFER}
  ${NVCAFFE_PARSER}
  ${NVINFER_PLUGIN}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDNN_LIBRARY}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  )

target_compile_options(tensorrt_apollo_cnn_lib
  PUBLIC
  -Wno-deprecated-declarations
  )

pybind11_add_module(${PROJECT_NAME}_pyb
  src/detector.cpp
  src/log_table.cpp
  src/feature_generator.cpp
  src/feature_map.cpp
  src/cluster2d.cpp)

target_link_libraries(${PROJECT_NAME}_pyb PRIVATE
  tensorrt_apollo_cnn_lib
  fmt::fmt
  )

set_target_properties(${PROJECT_NAME}_pyb PROPERTIES PREFIX "")               #  指定前缀
set_target_properties(${PROJECT_NAME}_pyb PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pyb)   #  指定文件名
set_target_properties(${PROJECT_NAME}_pyb PROPERTIES SUFFIX ".so")            #  指定后缀
set_target_properties(${PROJECT_NAME}_pyb PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/script")  # 指定库的输出路径
set_target_properties(${PROJECT_NAME}_pyb PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/script")  # 指定可执行文件的输出路径
