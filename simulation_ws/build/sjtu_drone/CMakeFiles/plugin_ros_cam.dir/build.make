# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/simulation_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/simulation_ws/build

# Include any dependencies generated for this target.
include sjtu_drone/CMakeFiles/plugin_ros_cam.dir/depend.make

# Include the progress variables for this target.
include sjtu_drone/CMakeFiles/plugin_ros_cam.dir/progress.make

# Include the compile flags for this target's objects.
include sjtu_drone/CMakeFiles/plugin_ros_cam.dir/flags.make

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/flags.make
sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o: /home/user/simulation_ws/src/sjtu_drone/src/plugin_ros_cam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user/simulation_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o -c /home/user/simulation_ws/src/sjtu_drone/src/plugin_ros_cam.cpp

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.i"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/simulation_ws/src/sjtu_drone/src/plugin_ros_cam.cpp > CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.i

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.s"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/simulation_ws/src/sjtu_drone/src/plugin_ros_cam.cpp -o CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.s

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.requires:

.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.requires

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.provides: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.requires
	$(MAKE) -f sjtu_drone/CMakeFiles/plugin_ros_cam.dir/build.make sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.provides.build
.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.provides

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.provides.build: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o


sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/flags.make
sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o: /home/user/simulation_ws/src/sjtu_drone/src/util_ros_cam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user/simulation_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o -c /home/user/simulation_ws/src/sjtu_drone/src/util_ros_cam.cpp

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.i"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/simulation_ws/src/sjtu_drone/src/util_ros_cam.cpp > CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.i

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.s"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/simulation_ws/src/sjtu_drone/src/util_ros_cam.cpp -o CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.s

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.requires:

.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.requires

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.provides: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.requires
	$(MAKE) -f sjtu_drone/CMakeFiles/plugin_ros_cam.dir/build.make sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.provides.build
.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.provides

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.provides.build: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o


# Object files for target plugin_ros_cam
plugin_ros_cam_OBJECTS = \
"CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o" \
"CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o"

# External object files for target plugin_ros_cam
plugin_ros_cam_EXTERNAL_OBJECTS =

/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/build.make
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libimage_transport.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libmessage_filters.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libclass_loader.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/libPocoFoundation.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libdl.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libroscpp.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libxmlrpcpp.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libroslib.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librospack.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libroscpp_serialization.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librostime.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libcpp_common.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_client.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_gui.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_sensors.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_rendering.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_physics.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_ode.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_transport.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_msgs.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_util.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_common.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_gimpact.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_opcode.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_opende_ou.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_math.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libprotobuf.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libsdformat.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libignition-math2.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgreMain.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgreTerrain.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgrePaging.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libignition-math2.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole_log4cxx.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librosconsole_backend_interface.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libxmlrpcpp.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libroslib.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librospack.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libroscpp_serialization.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/librostime.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /opt/ros/kinetic/lib/libcpp_common.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_client.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_gui.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_sensors.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_rendering.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_physics.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_ode.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_transport.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_msgs.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_util.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_common.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_gimpact.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_opcode.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_opende_ou.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/local/lib/libgazebo_math.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libprotobuf.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libsdformat.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgreMain.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgreTerrain.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: /usr/lib/x86_64-linux-gnu/libOgrePaging.so
/home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/user/simulation_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library /home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so"
	cd /home/user/simulation_ws/build/sjtu_drone && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/plugin_ros_cam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sjtu_drone/CMakeFiles/plugin_ros_cam.dir/build: /home/user/simulation_ws/src/sjtu_drone/plugins/libplugin_ros_cam.so

.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/build

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/requires: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/plugin_ros_cam.cpp.o.requires
sjtu_drone/CMakeFiles/plugin_ros_cam.dir/requires: sjtu_drone/CMakeFiles/plugin_ros_cam.dir/src/util_ros_cam.cpp.o.requires

.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/requires

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/clean:
	cd /home/user/simulation_ws/build/sjtu_drone && $(CMAKE_COMMAND) -P CMakeFiles/plugin_ros_cam.dir/cmake_clean.cmake
.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/clean

sjtu_drone/CMakeFiles/plugin_ros_cam.dir/depend:
	cd /home/user/simulation_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/simulation_ws/src /home/user/simulation_ws/src/sjtu_drone /home/user/simulation_ws/build /home/user/simulation_ws/build/sjtu_drone /home/user/simulation_ws/build/sjtu_drone/CMakeFiles/plugin_ros_cam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sjtu_drone/CMakeFiles/plugin_ros_cam.dir/depend

