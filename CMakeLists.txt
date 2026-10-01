cmake_minimum_required(VERSION 3.14) 

project(noise_reduction)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "C:/opt/opencv/build/x64/vc16/lib")
endif()

find_package(OpenCV REQUIRED)
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(FilterLib SpatialFilter.cpp)
target_link_libraries(FilterLib ${OpenCV_LIBS})

add_executable(noise_reduction main.cpp)
target_link_libraries(noise_reduction FilterLib)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest) 

enable_testing()

add_executable(FilterTests tests/test_filters.cpp)
target_link_libraries(FilterTests FilterLib gtest_main ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(FilterTests)