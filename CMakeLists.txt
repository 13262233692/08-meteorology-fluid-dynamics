cmake_minimum_required(VERSION 3.15)
project(MeteorologyFluidDynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math)
elseif(MSVC)
    add_compile_options(/O2 /fp:fast /EHsc /openmp:llvm /D_USE_MATH_DEFINES)
endif()

find_package(OpenMP REQUIRED)

find_package(netCDF QUIET)
if(netCDF_FOUND)
    message(STATUS "NetCDF found: ${netCDF_INCLUDE_DIRS}")
    add_definitions(-DMFD_HAVE_NETCDF)
else()
    message(STATUS "NetCDF NOT found - NetCDF reader will use synthetic mode only")
    set(netCDF_INCLUDE_DIRS "")
    set(netCDF_LIBRARIES "")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${netCDF_INCLUDE_DIRS}
)

set(SOURCES
    src/grid3d.cpp
    src/netcdf_reader.cpp
    src/fluid_solver.cpp
)

set(HEADERS
    include/grid3d.h
    include/netcdf_reader.h
    include/fluid_solver.h
)

add_library(mfd_core STATIC ${SOURCES} ${HEADERS})
target_link_libraries(mfd_core PUBLIC OpenMP::OpenMP_CXX ${netCDF_LIBRARIES})

pybind11_add_module(mfd_python src/pybind11_bindings.cpp)
target_link_libraries(mfd_python PRIVATE mfd_core)
set_target_properties(mfd_python PROPERTIES OUTPUT_NAME mfd)

enable_testing()
add_subdirectory(tests)
