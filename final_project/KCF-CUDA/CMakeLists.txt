# If running into cmake errors, change INCLUDE_DIRS and LIB_DIR to expected directories. Hardcoded so that we could begin writing in CUDA.

cmake_minimum_required(VERSION 2.8)
project(test)
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

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable( KCF ${sourcefiles} )
target_link_libraries( KCF ${OpenCV_LIBS})




