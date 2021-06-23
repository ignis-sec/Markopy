# CMakeList.txt : CMake project for Markopy, include source and define
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

file(GLOB_RECURSE MPY_FILES
     "src/Module/*.h"
     "src/Module/*.cpp"
)


if(WIN32)
    link_directories( $ENV{BOOST_ROOT}/stage/lib/)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)
    add_library(markopy MODULE ${MPY_FILES} )
else()
    add_library(markopy SHARED ${MPY_FILES} )
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)



find_package(Threads REQUIRED)
find_package(Boost COMPONENTS python${PYTHON_VER} REQUIRED)
find_package(PythonLibs REQUIRED)



set(MPY_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
set(MPY_LINK_FLAGS "${SGX_COMMON_CFLAGS} ")

set_target_properties(markopy PROPERTIES COMPILE_FLAGS "${MPY_CPP_FLAGS}")
set_target_properties(markopy PROPERTIES LINK_FLAGS "${MPY_LINK_FLAGS}")
set_target_properties(markopy PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(markopy PROPERTIES PREFIX "")
target_include_directories(markopy PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(markopy PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(markopy PRIVATE Boost::python${PYTHON_VER})
target_link_libraries(markopy PRIVATE Threads::Threads)
target_link_libraries(markopy PRIVATE markov)

if(WIN32)
    set_target_properties(markopy PROPERTIES SUFFIX ".pyd")
endif()