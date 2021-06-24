# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)


if (CMAKE_C_COMPILER MATCHES ".*mingw.*")
    message("Using ${CMAKE_CXX_COMPILER} to compile. CUDA code wont be compiled")
    project(Markopy LANGUAGES C CXX VERSION 0.5.1 DESCRIPTION "Password generation with markov chains")
else()
    message("Using ${CMAKE_CXX_COMPILER} to compile.")
    project(Markopy LANGUAGES C CXX VERSION 0.5.1 DESCRIPTION "Password generation with markov chains")
endif ()




# Include sub-projects.
add_subdirectory ("MarkovAPI")
#add_subdirectory ("CudaMarkovAPI")
add_subdirectory ("MarkovModel")
add_subdirectory ("MarkovAPICLI")
#add_subdirectory ("Markopy")
#add_subdirectory ("CudaMarkopy")
add_subdirectory ("MarkovPasswordsGUI")

 


# Global project options


#include(CTest)
