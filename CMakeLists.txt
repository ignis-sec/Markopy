cmake_minimum_required(VERSION 3.10)

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

##############################################################################################################
#####################################       MarkovModel project options      #################################
##############################################################################################################
#Header only library now




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


##############################################################################################################
#####################################     MarkovPassword project options     #################################
##############################################################################################################



file(GLOB_RECURSE MP_FILES
     "MarkovPasswords/src/*.h"
     "MarkovPasswords/src/*.cpp"
     "MarkovModel/src/*.h"
)

add_executable(markovp ${MP_FILES} )

set(MP_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")

set(MP_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(markovp PROPERTIES COMPILE_FLAGS "${MP_CPP_FLAGS}")
set_target_properties(markovp PROPERTIES LINK_FLAGS "${MP_LINK_FLAGS}")

set_target_properties(markovp PROPERTIES LINKER_LANGUAGE CXX)
find_package(Threads REQUIRED)
find_package(Boost 1.71.0 COMPONENTS program_options REQUIRED)

target_link_libraries(markovp Boost::program_options)
target_link_libraries(markovp Threads::Threads)
target_link_libraries(markovp markov)




##############################################################################################################
#####################################     Markopy project options            #################################
##############################################################################################################

file(GLOB_RECURSE MPY_FILES
     "Markopy/src/Module/*.h"
     "Markopy/src/Module/*.cpp"
)

add_library(markopy SHARED ${MPY_FILES} )
find_package(Threads REQUIRED)
find_package(Boost 1.71.0 COMPONENTS python REQUIRED)
find_package(PythonLibs REQUIRED)


set(MPY_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
set(MPY_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(markopy PROPERTIES COMPILE_FLAGS "${MPY_CPP_FLAGS}")
set_target_properties(markopy PROPERTIES LINK_FLAGS "${MPY_LINK_FLAGS}")
set_target_properties(markopy PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(markopy PROPERTIES PREFIX "")
target_include_directories(markopy PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(markopy ${PYTHON_LIBRARIES})
target_link_libraries(markopy Boost::python)
target_link_libraries(markopy Threads::Threads)
target_link_libraries(markopy markov)

##############################################################################################################
#####################################     CudaMarkovAPI project options      #################################
##############################################################################################################

file(GLOB_RECURSE CMP_FILES
     "CudaMarkovAPI/src/*.h"
     "CudaMarkovAPI/src/*.cu"
)

add_library(cudamarkov SHARED ${CMP_FILES})

find_package(Threads REQUIRED)

set(CMP_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
set(CMP_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(cudamarkov PROPERTIES COMPILE_FLAGS "${CMP_CPP_FLAGS}")
set_target_properties(cudamarkov PROPERTIES LINK_FLAGS "${CMP_LINK_FLAGS}")
set_target_properties(cudamarkov PROPERTIES LINKER_LANGUAGE CUDA)

set_target_properties(cudamarkov
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudamarkov Threads::Threads)
target_link_libraries(cudamarkov markov)

add_executable(cudamarkov_test CudaMarkovAPI/src/main.cu)
set_target_properties(cudamarkov_test PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(cudamarkov_test markov)
target_link_libraries(cudamarkov_test cudamarkov)


##############################################################################################################
#####################################     CudaMarkopy project options        #################################
##############################################################################################################

file(GLOB_RECURSE CMPY_FILES
     "CudaMarkopy/src/*.cu"
)

add_library(cudamarkopy SHARED ${CMPY_FILES})

find_package(Boost 1.71.0 COMPONENTS python REQUIRED)
find_package(PythonLibs REQUIRED)


set(CMPY_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
set(CMPY_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(cudamarkopy PROPERTIES COMPILE_FLAGS "${CMPY_CPP_FLAGS}")
set_target_properties(cudamarkopy PROPERTIES LINK_FLAGS "${CMPY_LINK_FLAGS}")

set_target_properties(cudamarkopy
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(cudamarkopy PROPERTIES PREFIX "")
target_include_directories(cudamarkopy PUBLIC ${PYTHON_INCLUDE_DIRS})

target_link_libraries(cudamarkopy cudamarkov)
target_link_libraries(cudamarkopy ${PYTHON_LIBRARIES})
target_link_libraries(cudamarkopy Boost::python)
target_link_libraries(cudamarkopy markov)
