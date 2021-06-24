# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

## BUILD EXAMPLES
# 
# Build everything:
# $ cmake . -DPYTHON_VER=38 && cmake --build .
# 
# This will build all the libraries and executables. Requires python-dev, CUDA, QT5, QT5-Webview
# 
# Build libraries only:
# $ cmake . -DPARTIAL=1 -DB_LIBS=1 && cmake --build .
# 
# Only build basic libraries. Requires only CXX compiler.
# 
# Build CUDA-accelerated libraries:
# $ cmake . -DPARTIAL=1 -DB_CUDA=1 && cmake --build .
# 
# Build libraries along with cuda accelerated ones.
# 
# Build python module & libraries:
# $ cmake . -DPARTIAL=1 -DPYTHON_VER=39 && cmake --build .
# 
# Will build basic libraries and python modules.
# 
# Build CUDA accelerated python module:
# $ cmake . -DPARTIAL=1 -DPYTHON_VER=39 -DB_CUDA=1 && cmake --build .
# 
# Will build cudamarkopy.
# 
# Build CUDA accelerated python module and the GUI:
# $ cmake . -DPARTIAL=1 -DPYTHON_VER=39 -DB_CUDA=1 -DB_GUI && cmake --build .
# 
# Combine methods
# 
##

if(NOT PROJ_VERSION)
    message("No version defined. Defaulting to 0.0.0")
    set(PROJ_VERSION "0.0.0")
endif()

message("Building version ${PROJ_VERSION}")

project(Markopy LANGUAGES CXX VERSION "${PROJ_VERSION}" DESCRIPTION "Password generation toolset with markov chains")
if(NOT PARTIAL)
    message("Building everything, including cuda")
    enable_language(CUDA)

    add_subdirectory ("MarkovAPI")
    add_subdirectory ("CudaMarkovAPI")
    add_subdirectory ("MarkovModel")
    add_subdirectory ("MarkovAPICLI")
    add_subdirectory ("Markopy")
    add_subdirectory ("CudaMarkopy")
    add_subdirectory ("MarkovPasswordsGUI")

else()
    if(B_LIBS)
        message("Building libraries")

        add_subdirectory ("MarkovAPI")
        add_subdirectory ("MarkovModel")
        add_subdirectory ("MarkovAPICLI")

    endif()
    if(B_CUDA)
        message("Building cuda libraries")
        enable_language(CUDA)

        add_subdirectory ("MarkovAPI")
        add_subdirectory ("MarkovModel")
        add_subdirectory ("CudaMarkovAPI")
    endif()
    if(PYTHON_VER)
    message("Building cpython extensions")

    add_subdirectory ("MarkovAPI")
    add_subdirectory ("MarkovModel")
    add_subdirectory ("Markopy")
    endif()
    if(B_CUDA)
        message("Building CUDA-accelerated cpython extensions")
        add_subdirectory ("CudaMarkopy")
    endif()
    if(B_GUI)
        message("Building GUI")

        add_subdirectory ("MarkovAPI")
        add_subdirectory ("MarkovModel")
        add_subdirectory ("MarkovPasswordsGUI")
    endif()
endif()

#include(CTest)
