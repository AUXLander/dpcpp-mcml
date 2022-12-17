cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

set(PROJECT_NAME dpcpp-mcml)
set(MCML_EMULATOR mcml-emulator)
set(MCML_GL_VIEW mcml-gl-view)

project(${PROJECT_NAME})

# Set default props for projects
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
	
	# Set DPC++ Compiler
	set(CMAKE_C_COMPILER icx)
	set(CMAKE_CXX_COMPILER icpx)
	
	# enable DPC++ snippet
	# find_package(IntelDPCPP REQUIRED)
	
else()

	# Set llvm sycl
	set(CMAKE_C_COMPILER ~/sycl/llvm/build/bin/clang++)
	set(CMAKE_CXX_COMPILER ~/sycl/llvm/build/bin/clang++)

endif(WIN32)

# add executable MCML_EMULATOR
add_executable(
	${MCML_EMULATOR}
	
    src/main.cpp 
    src/mcml.hpp
    src/matrix.hpp
	src/iofile.hpp
	src/pipe.hpp
)


