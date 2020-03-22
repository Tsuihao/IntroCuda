############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files
cmake_minimum_required(VERSION 3.16)
project(cuda_rgba2grey LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

INCLUDE(FindCUDA)

# CUDA include directories
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

# OpenCV include directories
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

set(HW1_files
    main.cpp 
    reference_calc.cpp 
    compare.cpp)


set(SOURCES
   HW1.cpp
   ${HW1_files}
   ${hdr}
   ${cu})

set(LIBRARIES
   ${OpenCV_LIBS})

message("${CMAKE_MODULE_PATH}")

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRAEIS})