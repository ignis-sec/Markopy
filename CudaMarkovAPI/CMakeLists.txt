
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