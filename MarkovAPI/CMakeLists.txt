# CMakeList.txt : CMake project for MarkovAPI, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


# TODO: Add tests and install targets if needed.


##############################################################################################################
#####################################       MarkovAPI project options        #################################
##############################################################################################################


file(GLOB_RECURSE MPLIB_FILES
     "MarkovAPI/src/*.h"
     "MarkovAPI/src/*.cpp"
     "MarkovModel/src/*.h"
)

add_library(markov SHARED ${MPLIB_FILES} )

set(MPLIB_CPP_FLAGS "${MPLIB_C_FLAGS} -std=c++17")

set(MPLIB_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(markov PROPERTIES COMPILE_FLAGS "${MPLIB_CPP_FLAGS}")
set_target_properties(markov PROPERTIES LINK_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPLIB_LINK_FLAGS}")
set_target_properties(markov PROPERTIES LINKER_LANGUAGE CXX)
