set(CMAKE_CROSSCOMPILING TRUE)
SET(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Clang target triple
SET(TARGET armv7-none-eabi)

# specify the cross compiler
SET(CMAKE_C_COMPILER_TARGET ${TARGET})
SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_COMPILER_TARGET ${TARGET})
SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_ASM_COMPILER_TARGET ${TARGET})
SET(CMAKE_ASM_COMPILER clang)

# C/C++ toolchain
SET(TOOLCHAIN "C:/Program Files (x86)/GNU Tools ARM Embedded/7 2018-q2-update")
SET(CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN ${TOOLCHAIN})
SET(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN ${TOOLCHAIN})

# specify compiler flags
SET(ARCH_FLAGS "-target armv7-none-eabi -mcpu=cortex-a5")
SET(CMAKE_C_FLAGS "-Wall -Wextra ${ARCH_FLAGS}" CACHE STRING "Common flags for C compiler")
SET(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -fno-exceptions -fno-threadsafe-statics ${ARCH_FLAGS}" CACHE STRING "Common flags for C++ compiler")
