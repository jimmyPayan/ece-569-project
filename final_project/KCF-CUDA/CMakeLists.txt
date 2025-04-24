cmake_minimum_required(VERSION 3.8)
project(test LANGUAGES CXX CUDA)

# OpenCV
set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../installOpenCV/include")
set(OpenCV_LIB_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/../installOpenCV/lib")
set(OpenCV_LIBS
    ${OpenCV_LIB_DIR}/libopencv_core.so
    ${OpenCV_LIB_DIR}/libopencv_highgui.so
    ${OpenCV_LIB_DIR}/libopencv_imgproc.so
    ${OpenCV_LIB_DIR}/libopencv_imgcodecs.so.3.0
    ${OpenCV_LIB_DIR}/libopencv_videoio.so.3.0
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  src
  src/fhog
)
link_directories(${OpenCV_LIB_DIR})

# Compiler settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Sources
file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CU_SOURCES  src/*.cu)

# Executable
add_executable(KCF ${CPP_SOURCES} ${CU_SOURCES})

set_target_properties(KCF PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 11
)

target_link_libraries(KCF
  PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)
