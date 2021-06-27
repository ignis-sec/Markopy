# CMakeList.txt : CMake project for MarkovAPI, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})
link_directories(${PROJECT_SOURCE_DIR}/out/lib)



file(GLOB_RECURSE MPLIB_FILES
     "src/*.h"
     "src/*.cpp"
     "../MarkovModel/src/*.h"
)
if (WIN32)
    add_library(markov STATIC ${MPLIB_FILES} )
else()
    add_library(markov SHARED ${MPLIB_FILES} )
endif()

set(MPLIB_CPP_FLAGS "${MPLIB_C_FLAGS} -std=c++17")
if(WIN32)
set(MPLIB_LINK_FLAGS "${SGX_COMMON_CFLAGS}")
else()
set(MPLIB_LINK_FLAGS "${SGX_COMMON_CFLAGS} -Wl,-rpath='${ORIGIN}'")
endif()

set_target_properties(markov PROPERTIES COMPILE_FLAGS "${MPLIB_CPP_FLAGS}")
set_target_properties(markov PROPERTIES LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPLIB_LINK_FLAGS}")
set_target_properties(markov PROPERTIES LINKER_LANGUAGE CXX)
