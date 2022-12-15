cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

set(PROJECT_NAME dpcpp-mcml)
set(MCML_EMULATOR mcml-emulator)
set(MCML_GL_VIEW mcml-gl-view)

project(${PROJECT_NAME})

# Set DPC++ Compiler
#set(CMAKE_C_COMPILER icx)
#set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_C_COMPILER ~/sycl/llvm/build/bin/clang++)
set(CMAKE_CXX_COMPILER ~/sycl/llvm/build/bin/clang++)

# Set default props for projects
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# enable DPC++ snippet
# find_package(IntelDPCPP REQUIRED)


# add executable MCML_EMULATOR
add_executable(
	${MCML_EMULATOR}
 
      src/emulator/main.cpp 
      src/emulator/mcml.hpp
      src/common/matrix.hpp
	src/common/iofile.hpp
	src/common/pipe.hpp
)


