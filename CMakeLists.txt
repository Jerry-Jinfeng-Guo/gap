cmake_minimum_required(VERSION 3.18)



# Check if CUDA is disabled
if(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit)
    # Project without CUDA
    project(GAP VERSION 1.0.0 LANGUAGES C CXX)
    set(GAP_ENABLE_CUDA OFF)
else()
    # Try to enable CUDA, fallback to C++ only if not available
    
    # Set CUDA compiler from environment if available
    if(DEFINED ENV{CUDACXX} AND NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})
        message(STATUS "Using CUDA compiler from environment: ${CMAKE_CUDA_COMPILER}")
    endif()
    
    # Debug: Show what we're working with
    message(STATUS "Environment CUDACXX: $ENV{CUDACXX}")
    message(STATUS "Environment CUDA_HOME: $ENV{CUDA_HOME}")
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        project(GAP VERSION 1.0.0 LANGUAGES C CXX CUDA)
        set(GAP_ENABLE_CUDA ON)
        message(STATUS "CUDA enabled - Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    else()
        project(GAP VERSION 1.0.0 LANGUAGES C CXX)
        set(GAP_ENABLE_CUDA OFF)
        message(STATUS "CUDA not found, building CPU-only version")
    endif()
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find CUDA packages only if enabled
if(GAP_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA compiler flags for compatibility and warning reduction
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    
    # Reduce nvlink warnings by preferring shared libraries and suppressing warnings
    set(CMAKE_CUDA_DEVICE_LINK_FLAGS "-Wno-deprecated-gpu-targets")
    
    # Suppress nvlink warnings about incompatible static libraries
    set(CMAKE_CUDA_DEVICE_LINK_FLAGS "${CMAKE_CUDA_DEVICE_LINK_FLAGS} --suppress-link-warnings")
    
    # CMake will automatically handle host compiler setting via CMAKE_CUDA_HOST_COMPILER
    
    # Define macro to enable CUDA availability check in runtime
    add_compile_definitions(GAP_CUDA_AVAILABLE=1)
    message(STATUS "CUDA enabled - GAP_CUDA_AVAILABLE defined")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
    add_compile_options(/permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# CUDA-specific options (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    # Set CUDA architecture - auto-detect or use compatible defaults
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Try to detect GPU compute capability, fallback to common architectures
        set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")
        message(STATUS "Using multiple CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --extended-lambda --expt-relaxed-constexpr")
    # Suppress clang-format line directive warnings and other style warnings 
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
endif()

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Optional: Install rules
install(TARGETS gap_main
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/gap/
    DESTINATION include/gap
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)
