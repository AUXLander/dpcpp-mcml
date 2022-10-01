cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

set(PROJECT_NAME dpcpp-mcml)

project(${PROJECT_NAME})

# Set DPC++ Compiler
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# Set default props for projects
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# enable DPC++ snippet
# find_package(IntelDPCPP REQUIRED)

add_executable(
	${PROJECT_NAME}
 
      src/main.cpp 
      src/mcml.hpp
      src/matrix.hpp
)
