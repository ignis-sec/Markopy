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

link_directories(${PROJECT_SOURCE_DIR}/out/lib)

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE CMPY_FILES
     "src/*.cu"
)

if(WIN32)
    link_directories( $ENV{BOOST_ROOT}/stage/lib/)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

string(REGEX REPLACE "^(.)(.+)" "\\1.\\2" PYTHON_VER_SEP ${PYTHON_VER})

find_package(Threads REQUIRED)
find_package(Boost COMPONENTS python${PYTHON_VER} REQUIRED)
if(PY_DEV_PATH)
    message("Using ${PY_DEV_PATH}")
    include_directories("${PY_DEV_PATH}/include")
    link_directories("${PY_DEV_PATH}/libs")
else()
    find_package(PythonLibs ${PYTHON_VER_SEP} REQUIRED)
endif()


add_library(cudamarkopy SHARED ${CMPY_FILES})
add_dependencies (cudamarkopy cudamarkov)

set(CMPY_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
if(WIN32)
set(CMPY_LINK_FLAGS "${SGX_COMMON_CFLAGS}")
else()
set(CMPY_LINK_FLAGS "${SGX_COMMON_CFLAGS} -Wl,-rpath='${ORIGIN}'")
endif()


set_target_properties(cudamarkopy PROPERTIES COMPILE_FLAGS "${CMPY_CPP_FLAGS}")
set_target_properties(cudamarkopy PROPERTIES LINK_FLAGS "${CMPY_LINK_FLAGS}")

set_target_properties(cudamarkopy
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(cudamarkopy PROPERTIES PREFIX "")
target_include_directories(cudamarkopy PUBLIC ${PYTHON_INCLUDE_DIRS})

target_link_libraries(cudamarkopy cudamarkov)
target_link_libraries(cudamarkopy ${PYTHON_LIBRARIES})
target_link_libraries(cudamarkopy Boost::python${PYTHON_VER})
target_link_libraries(cudamarkopy markov)
if(WIN32)
    set_target_properties(markopy PROPERTIES SUFFIX ".pyd")
endif()