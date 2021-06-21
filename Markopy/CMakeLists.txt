
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