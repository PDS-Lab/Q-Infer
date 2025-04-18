# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /home/wq/.local/lib/python2.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/wq/.local/lib/python2.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wq/Q-infer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wq/Q-infer/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test-tokenizer-1-llama.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-tokenizer-1-llama.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-tokenizer-1-llama.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-tokenizer-1-llama.dir/flags.make

tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o: tests/CMakeFiles/test-tokenizer-1-llama.dir/flags.make
tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o: /home/wq/Q-infer/tests/test-tokenizer-1-llama.cpp
tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o: tests/CMakeFiles/test-tokenizer-1-llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wq/Q-infer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o"
	cd /home/wq/Q-infer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o -MF CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o.d -o CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o -c /home/wq/Q-infer/tests/test-tokenizer-1-llama.cpp

tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.i"
	cd /home/wq/Q-infer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wq/Q-infer/tests/test-tokenizer-1-llama.cpp > CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.i

tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.s"
	cd /home/wq/Q-infer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wq/Q-infer/tests/test-tokenizer-1-llama.cpp -o CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.s

# Object files for target test-tokenizer-1-llama
test__tokenizer__1__llama_OBJECTS = \
"CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o"

# External object files for target test-tokenizer-1-llama
test__tokenizer__1__llama_EXTERNAL_OBJECTS =

bin/test-tokenizer-1-llama: tests/CMakeFiles/test-tokenizer-1-llama.dir/test-tokenizer-1-llama.cpp.o
bin/test-tokenizer-1-llama: tests/CMakeFiles/test-tokenizer-1-llama.dir/build.make
bin/test-tokenizer-1-llama: libllama.a
bin/test-tokenizer-1-llama: common/libcommon.a
bin/test-tokenizer-1-llama: libllama.a
bin/test-tokenizer-1-llama: /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so
bin/test-tokenizer-1-llama: /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so
bin/test-tokenizer-1-llama: /usr/local/cuda-12.2/targets/x86_64-linux/lib/libculibos.a
bin/test-tokenizer-1-llama: /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so
bin/test-tokenizer-1-llama: tests/CMakeFiles/test-tokenizer-1-llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wq/Q-infer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-tokenizer-1-llama"
	cd /home/wq/Q-infer/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-tokenizer-1-llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-tokenizer-1-llama.dir/build: bin/test-tokenizer-1-llama
.PHONY : tests/CMakeFiles/test-tokenizer-1-llama.dir/build

tests/CMakeFiles/test-tokenizer-1-llama.dir/clean:
	cd /home/wq/Q-infer/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-tokenizer-1-llama.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-tokenizer-1-llama.dir/clean

tests/CMakeFiles/test-tokenizer-1-llama.dir/depend:
	cd /home/wq/Q-infer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wq/Q-infer /home/wq/Q-infer/tests /home/wq/Q-infer/build /home/wq/Q-infer/build/tests /home/wq/Q-infer/build/tests/CMakeFiles/test-tokenizer-1-llama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test-tokenizer-1-llama.dir/depend

