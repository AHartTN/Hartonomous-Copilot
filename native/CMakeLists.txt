cmake_minimum_required(VERSION 3.20)
project(hartonomous_native VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(Eigen3 3.4 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenMP)

# Source files
set(SOURCES
    src/hilbert.cpp
    src/manifold.cpp
    src/utils.cpp
    src/stats.cpp
    src/version.cpp
)

# Shared library
add_library(hartonomous_native SHARED ${SOURCES})

target_include_directories(hartonomous_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hartonomous_native
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ZLIB::ZLIB
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hartonomous_native PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for parallelization")
else()
    message(WARNING "OpenMP not found - parallel operations will be slower")
endif()

# Compiler flags
if(MSVC)
    target_compile_options(hartonomous_native PRIVATE
        /W4          # Warning level 4
        /O2          # Optimize for speed
        /arch:AVX2   # Use AVX2 if available
    )
else()
    target_compile_options(hartonomous_native PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
        -march=native
        -ffast-math
    )
endif()

# Export symbols
set_target_properties(hartonomous_native PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/hartonomous_native.h
)

if(WIN32)
    target_compile_definitions(hartonomous_native PRIVATE HARTONOMOUS_EXPORTS)
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS hartonomous_native
    EXPORT hartonomous_nativeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install to Python site-packages for easy access
if(DEFINED PYTHON_SITE_PACKAGES)
    install(TARGETS hartonomous_native
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/hartonomous
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}/hartonomous
    )
endif()

# Install to PostgreSQL lib directory for C extension
if(DEFINED POSTGRESQL_LIB_DIR)
    install(TARGETS hartonomous_native
        LIBRARY DESTINATION ${POSTGRESQL_LIB_DIR}
        RUNTIME DESTINATION ${POSTGRESQL_LIB_DIR}
    )
endif()

# Export targets
install(EXPORT hartonomous_nativeTargets
    FILE hartonomous_nativeTargets.cmake
    NAMESPACE hartonomous::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hartonomous_native
)

# Testing
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration
message(STATUS "")
message(STATUS "Hartonomous Native Configuration:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen:        ${EIGEN3_VERSION}")
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
message(STATUS "")
