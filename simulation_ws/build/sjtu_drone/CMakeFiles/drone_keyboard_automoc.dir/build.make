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

# Utility rule file for drone_keyboard_automoc.

# Include the progress variables for this target.
include sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/progress.make

sjtu_drone/CMakeFiles/drone_keyboard_automoc:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/user/simulation_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic moc for target drone_keyboard"
	cd /home/user/simulation_ws/build/sjtu_drone && /usr/bin/cmake -E cmake_autogen /home/user/simulation_ws/build/sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/ ""

drone_keyboard_automoc: sjtu_drone/CMakeFiles/drone_keyboard_automoc
drone_keyboard_automoc: sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/build.make

.PHONY : drone_keyboard_automoc

# Rule to build all files generated by this target.
sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/build: drone_keyboard_automoc

.PHONY : sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/build

sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/clean:
	cd /home/user/simulation_ws/build/sjtu_drone && $(CMAKE_COMMAND) -P CMakeFiles/drone_keyboard_automoc.dir/cmake_clean.cmake
.PHONY : sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/clean

sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/depend:
	cd /home/user/simulation_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/simulation_ws/src /home/user/simulation_ws/src/sjtu_drone /home/user/simulation_ws/build /home/user/simulation_ws/build/sjtu_drone /home/user/simulation_ws/build/sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sjtu_drone/CMakeFiles/drone_keyboard_automoc.dir/depend

