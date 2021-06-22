# CMakeList.txt : CMake project for MarkovPasswordsGUI, include source and define
# project specific logic here.
#

cmake_minimum_required (VERSION 3.8)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories( ${CMAKE_BINARY_DIR}/out/lib)



file(GLOB_RECURSE MPGUI_UI
     "*.ui"
     "src/*.ui"
)


find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt5 COMPONENTS WebEngineWidgets  REQUIRED)
find_package(Qt5 COMPONENTS Core REQUIRED)

qt5_wrap_ui(UIS_HDRS ${MPGUI_UI})

file(GLOB_RECURSE MPGUI_FILES
     "src/*.cpp"
     "src/*.h"
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)



add_executable(MPGUI ${MPGUI_FILES} ${UIS_HDRS})
target_link_libraries(MPGUI markov)
target_link_libraries(MPGUI Qt5::Widgets)
target_link_libraries(MPGUI Qt5::Core)
target_link_libraries(MPGUI Qt5::WebEngineWidgets )








