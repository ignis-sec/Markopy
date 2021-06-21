# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project(Markopy LANGUAGES C CXX CUDA VERSION 0.5.1 DESCRIPTION "Password generation with markov chains")

# Include sub-projects.
add_subdirectory ("MarkovAPI")
add_subdirectory ("CudaMarkovAPI")
add_subdirectory ("MarkovModel")
add_subdirectory ("MarkovAPICLI")
add_subdirectory ("Markopy")
add_subdirectory ("CudaMarkopy")
add_subdirectory ("MarkovPasswordsGUI")

 


# Global project options


#include(CTest)
