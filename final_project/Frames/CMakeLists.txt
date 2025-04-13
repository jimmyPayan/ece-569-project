cmake_minimum_required(VERSION 2.8)
project(test)

set(OpenCV_INCLUDE_DIRS "$ENV{HOME}/ece569/final-project/opencv-install/include")
set(OpenCV_LIB_DIR "$ENV{HOME}/ece569/final-project/opencv-install/lib")

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
add_executable( Frames ${sourcefiles} )
target_link_libraries( Frames ${OpenCV_LIBS})




