# CMakeList.txt : CMake project for MarkovAPICLI, include source and define
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


set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)

file(GLOB_RECURSE MP_FILES
     "src/*.h"
     "src/*.cpp"
)

if(WIN32)
    link_directories( $ENV{BOOST_ROOT}/stage/lib/)
endif()

add_executable(markovp ${MP_FILES} )

set(MP_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")

set(MP_LINK_FLAGS "${SGX_COMMON_CFLAGS}")

set_target_properties(markovp PROPERTIES COMPILE_FLAGS "${MP_CPP_FLAGS}")
set_target_properties(markovp PROPERTIES LINK_FLAGS "${MP_LINK_FLAGS}")

set_target_properties(markovp PROPERTIES LINKER_LANGUAGE CXX)
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(markovp Boost::program_options)
target_link_libraries(markovp Threads::Threads)
target_link_libraries(markovp markov)


