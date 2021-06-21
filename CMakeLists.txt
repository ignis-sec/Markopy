# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("Markopy")

# Include sub-projects.
add_subdirectory ("MarkovAPI")
add_subdirectory ("CudaMarkovAPI")
add_subdirectory ("MarkovModel")
add_subdirectory ("MarkovAPICLI")
add_subdirectory ("Markopy")
add_subdirectory ("CudaMarkopy")
add_subdirectory ("MarkovPasswordsGUI")


# Global project options
project(model LANGUAGES C CXX CUDA VERSION 0.5.1 DESCRIPTION "MarkovModel shared Library")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/lib)
set(THREADS_PREFER_PTHREAD_FLAG ON)
include(CTest)
