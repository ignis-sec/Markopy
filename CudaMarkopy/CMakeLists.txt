# CMakeList.txt : CMake project for CudaMarkopy, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})
link_directories( ${CMAKE_BINARY_DIR}/out/lib)

if(WIN32)
    link_directories( $ENV{BOOST_ROOT}/stage/lib/)
endif()

file(GLOB_RECURSE CMPY_FILES
     "src/*.cu"
)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

add_library(cudamarkopy SHARED ${CMPY_FILES})

find_package(Boost COMPONENTS python38 REQUIRED)
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
target_link_libraries(cudamarkopy Boost::python38)
target_link_libraries(cudamarkopy markov)
if(WIN32)
    set_target_properties(markopy PROPERTIES SUFFIX ".pyd")
endif()