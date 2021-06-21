


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
