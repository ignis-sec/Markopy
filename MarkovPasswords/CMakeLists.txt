
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

