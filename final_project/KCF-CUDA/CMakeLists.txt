# If running into cmake errors, change INCLUDE_DIRS and LIB_DIR to expected directories. Hardcoded so that we could begin writing in CUDA.

cmake_minimum_required(VERSION 3.8)
project(KCF LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../installOpenCV/include")
#set(OpenCV_LIB_DIR "$ENV{HOME}/ece569/final_project/installOpenCV/lib")
set(OpenCV_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../installOpenCV/lib")
#set(OpenCV_LIB_DIR "$ENV{HOME}/ece-569-project/final_project/installOpenCV/lib")

set(OpenCV_LIBS 
	${OpenCV_LIB_DIR}/libopencv_core.so
	${OpenCV_LIB_DIR}/libopencv_highgui.so
	${OpenCV_LIB_DIR}/libopencv_imgproc.so
	${OpenCV_LIB_DIR}/libopencv_imgcodecs.so.3.0
	${OpenCV_LIB_DIR}/libopencv_videoio.so.3.0
)


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
include_directories(src)
include_directories(/opt/ohpc/pub/apps/cuda/11.0/include)

set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

if(NOT WIN32)
	add_definitions("-O3 -fopenmp")
endif()

FILE(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
set(CUDA_SOURCES src/gaussianCorrelation.cu src/normalizeAndTruncate.cu)
#Mark CUDA files
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
#Add executable
add_executable( KCF ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries( KCF ${OpenCV_LIBS} cuda cudart cufft)



