# CMakeList.txt : CMake project for CudaMarkovAPI, include source and define
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


file(GLOB_RECURSE CMP_FILES
     "src/*.h"
     "src/*.cu"
)

if(WIN32)
add_library(cudamarkov STATIC ${CMP_FILES} src/cudaDeviceController.cu src/cudaModelMatrix.cu)
else()
    add_library(cudamarkov SHARED ${CMP_FILES})
endif()

add_dependencies (cudamarkov markov)

set(CMP_CPP_FLAGS "${MP_C_FLAGS} -std=c++17")
set(CMP_LINK_FLAGS "${SGX_COMMON_CFLAGS} -fPIC")

set_target_properties(cudamarkov PROPERTIES COMPILE_FLAGS "${CMP_CPP_FLAGS}")
set_target_properties(cudamarkov PROPERTIES LINK_FLAGS "${CMP_LINK_FLAGS}")
set_target_properties(cudamarkov PROPERTIES LINKER_LANGUAGE CUDA)

set_target_properties(cudamarkov
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudamarkov markov)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/bin)
add_executable(cudamarkovp src/main.cu)
add_dependencies (cudamarkovp cudamarkov)
set_target_properties(cudamarkovp PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(cudamarkovp PROPERTIES COMPILE_FLAGS "${CMP_CPP_FLAGS}")
set_target_properties(cudamarkovp PROPERTIES LINK_FLAGS "${CMP_LINK_FLAGS}")
target_link_libraries(cudamarkovp markov)
target_link_libraries(cudamarkovp cudamarkov)