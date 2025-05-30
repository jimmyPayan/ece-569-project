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

# Utility rule file for pch_Generate_opencv_videoio.

# Include any custom commands dependencies for this target.
include modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/compiler_depend.make

# Include the progress variables for this target.
include modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/progress.make

modules/videoio/CMakeFiles/pch_Generate_opencv_videoio: modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch

modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch: /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio/src/precomp.hpp
modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch: modules/videoio/precomp.hpp
modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch: lib/libopencv_videoio_pch_dephelp.a
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/u20/jimmypayan/ece569/final-project/build-openCV/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating precomp.hpp.gch/opencv_videoio_Release.gch"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio && /opt/ohpc/pub/apps/cmake/3.28.3/bin/cmake -E make_directory /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio/precomp.hpp.gch
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio && /bin/c++ -O3 -DNDEBUG -DNDEBUG -fPIC -DCVAPI_EXPORTS -isystem"/home/u20/jimmypayan/ece569/final-project/build-openCV" -isystem"/usr/include/gtk-3.0" -isystem"/usr/include/atk-1.0" -isystem"/usr/include/at-spi2-atk/2.0" -isystem"/usr/include" -isystem"/usr/include/pango-1.0" -isystem"/usr/include/gio-unix-2.0" -isystem"/usr/include/cairo" -isystem"/usr/include/gdk-pixbuf-2.0" -isystem"/usr/include/glib-2.0" -isystem"/usr/lib64/glib-2.0/include" -isystem"/usr/include/at-spi-2.0" -isystem"/usr/include/dbus-1.0" -isystem"/usr/lib64/dbus-1.0/include" -isystem"/usr/include/libdrm" -isystem"/usr/include/harfbuzz" -isystem"/usr/include/freetype2" -isystem"/usr/include/fribidi" -isystem"/usr/include/libpng15" -isystem"/usr/include/uuid" -isystem"/usr/include/pixman-1" -isystem"/usr/include/glib-2.0" -isystem"/usr/lib64/glib-2.0/include" -isystem"/usr/include" -isystem"/home/u20/jimmypayan/ece569/final-project/build-openCV" -isystem"/usr/include/gtk-3.0" -isystem"/usr/include/atk-1.0" -isystem"/usr/include/at-spi2-atk/2.0" -isystem"/usr/include" -isystem"/usr/include/pango-1.0" -isystem"/usr/include/gio-unix-2.0" -isystem"/usr/include/cairo" -isystem"/usr/include/gdk-pixbuf-2.0" -isystem"/usr/include/glib-2.0" -isystem"/usr/lib64/glib-2.0/include" -isystem"/usr/include/at-spi-2.0" -isystem"/usr/include/dbus-1.0" -isystem"/usr/lib64/dbus-1.0/include" -isystem"/usr/include/libdrm" -isystem"/usr/include/harfbuzz" -isystem"/usr/include/freetype2" -isystem"/usr/include/fribidi" -isystem"/usr/include/libpng15" -isystem"/usr/include/uuid" -isystem"/usr/include/pixman-1" -isystem"/usr/include/glib-2.0" -isystem"/usr/lib64/glib-2.0/include" -isystem"/usr/include" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio/include" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio/src" -isystem"/home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/hal/include" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/core/include" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/imgproc/include" -I"/home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/imgcodecs/include" -D__OPENCV_BUILD=1 -D__OPENCV_BUILD=1 -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wno-narrowing -Wno-delete-non-virtual-dtor -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -DCVAPI_EXPORTS -x c++-header -o /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio/precomp.hpp

modules/videoio/precomp.hpp: /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio/src/precomp.hpp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/u20/jimmypayan/ece569/final-project/build-openCV/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating precomp.hpp"
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio && /opt/ohpc/pub/apps/cmake/3.28.3/bin/cmake -E copy /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio/src/precomp.hpp /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio/precomp.hpp

pch_Generate_opencv_videoio: modules/videoio/CMakeFiles/pch_Generate_opencv_videoio
pch_Generate_opencv_videoio: modules/videoio/precomp.hpp
pch_Generate_opencv_videoio: modules/videoio/precomp.hpp.gch/opencv_videoio_Release.gch
pch_Generate_opencv_videoio: modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/build.make
.PHONY : pch_Generate_opencv_videoio

# Rule to build all files generated by this target.
modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/build: pch_Generate_opencv_videoio
.PHONY : modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/build

modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/clean:
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio && $(CMAKE_COMMAND) -P CMakeFiles/pch_Generate_opencv_videoio.dir/cmake_clean.cmake
.PHONY : modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/clean

modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/depend:
	cd /home/u20/jimmypayan/ece569/final-project/build-openCV && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0 /home/u20/jimmypayan/ece569/final-project/opencv-3.0.0/modules/videoio /home/u20/jimmypayan/ece569/final-project/build-openCV /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio /home/u20/jimmypayan/ece569/final-project/build-openCV/modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : modules/videoio/CMakeFiles/pch_Generate_opencv_videoio.dir/depend

