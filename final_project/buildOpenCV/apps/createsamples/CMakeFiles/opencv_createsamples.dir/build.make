# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/ohpc/pub/apps/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /opt/ohpc/pub/apps/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u20/jimmypayan/ece569/final-project/build-openCV

# Include any dependencies generated for this target.
include apps/createsamples/CMakeFiles/opencv_createsamples.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include apps/createsamples/CMakeFiles/opencv_createsamples.dir/compiler_depend.make

# Include the progress variables for this target.
include apps/createsamples/CMakeFiles/opencv_createsamples.dir/progress.make

# Include the compile flags for this target's objects.
include apps/createsamples/CMakeFiles/opencv_createsamples.dir/flags.make

apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o: apps/createsamples/CMakeFiles/opencv_createsamples.dir/flags.make
apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o: /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/createsamples.cpp
apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o: apps/createsamples/CMakeFiles/opencv_createsamples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/u20/jimmypayan/ece569/final-project/build-openCV/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o -MF CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o.d -o CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o -c /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/createsamples.cpp

apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv_createsamples.dir/createsamples.cpp.i"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/createsamples.cpp > CMakeFiles/opencv_createsamples.dir/createsamples.cpp.i

apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv_createsamples.dir/createsamples.cpp.s"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/createsamples.cpp -o CMakeFiles/opencv_createsamples.dir/createsamples.cpp.s

apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o: apps/createsamples/CMakeFiles/opencv_createsamples.dir/flags.make
apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o: /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/utility.cpp
apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o: apps/createsamples/CMakeFiles/opencv_createsamples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/u20/jimmypayan/ece569/final-project/build-openCV/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o -MF CMakeFiles/opencv_createsamples.dir/utility.cpp.o.d -o CMakeFiles/opencv_createsamples.dir/utility.cpp.o -c /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/utility.cpp

apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv_createsamples.dir/utility.cpp.i"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/utility.cpp > CMakeFiles/opencv_createsamples.dir/utility.cpp.i

apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv_createsamples.dir/utility.cpp.s"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples/utility.cpp -o CMakeFiles/opencv_createsamples.dir/utility.cpp.s

# Object files for target opencv_createsamples
opencv_createsamples_OBJECTS = \
"CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o" \
"CMakeFiles/opencv_createsamples.dir/utility.cpp.o"

# External object files for target opencv_createsamples
opencv_createsamples_EXTERNAL_OBJECTS =

bin/opencv_createsamples: apps/createsamples/CMakeFiles/opencv_createsamples.dir/createsamples.cpp.o
bin/opencv_createsamples: apps/createsamples/CMakeFiles/opencv_createsamples.dir/utility.cpp.o
bin/opencv_createsamples: apps/createsamples/CMakeFiles/opencv_createsamples.dir/build.make
bin/opencv_createsamples: lib/libopencv_objdetect.so.3.0.0
bin/opencv_createsamples: lib/libopencv_calib3d.so.3.0.0
bin/opencv_createsamples: lib/libopencv_features2d.so.3.0.0
bin/opencv_createsamples: lib/libopencv_highgui.so.3.0.0
bin/opencv_createsamples: lib/libopencv_videoio.so.3.0.0
bin/opencv_createsamples: lib/libopencv_imgcodecs.so.3.0.0
bin/opencv_createsamples: lib/libopencv_imgproc.so.3.0.0
bin/opencv_createsamples: lib/libopencv_ml.so.3.0.0
bin/opencv_createsamples: lib/libopencv_flann.so.3.0.0
bin/opencv_createsamples: lib/libopencv_core.so.3.0.0
bin/opencv_createsamples: lib/libopencv_hal.a
bin/opencv_createsamples: apps/createsamples/CMakeFiles/opencv_createsamples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/u20/jimmypayan/ece569/final-project/build-openCV/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../bin/opencv_createsamples"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_createsamples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/createsamples/CMakeFiles/opencv_createsamples.dir/build: bin/opencv_createsamples
.PHONY : apps/createsamples/CMakeFiles/opencv_createsamples.dir/build

apps/createsamples/CMakeFiles/opencv_createsamples.dir/clean:
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples && $(CMAKE_COMMAND) -P CMakeFiles/opencv_createsamples.dir/cmake_clean.cmake
.PHONY : apps/createsamples/CMakeFiles/opencv_createsamples.dir/clean

apps/createsamples/CMakeFiles/opencv_createsamples.dir/depend:
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0 /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/apps/createsamples /home/u20/jimmypayan/ece569/final-project/build-openCV /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples /home/u20/jimmypayan/ece569/final-project/build-openCV/apps/createsamples/CMakeFiles/opencv_createsamples.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : apps/createsamples/CMakeFiles/opencv_createsamples.dir/depend

