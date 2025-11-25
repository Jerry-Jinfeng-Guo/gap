# CMakeLists.txt for GAP Python Bindings
# Handles both CPU and GPU (CUDA) backends with conditional compilation

cmake_minimum_required(VERSION 3.18)

# Python module target with conditional CUDA support
set(GAP_PYTHON_SOURCES
    gap_bindings.cpp
    core_bindings.cpp
    solver_bindings.cpp
    io_bindings.cpp
)

# Add CUDA sources only if CUDA is available
# NOTE: GPU backend currently disabled for Python bindings (see linking section below)
# GAP_CUDA_AVAILABLE is already defined globally in main CMakeLists.txt
if(GAP_ENABLE_CUDA AND FALSE)  # Temporarily disabled
    list(APPEND GAP_PYTHON_SOURCES cuda_bindings.cpp)
    message(STATUS "Python bindings: CUDA support enabled")
else()
    message(STATUS "Python bindings: CPU-only build (GPU backend available in C++)")
endif()

# Create the Python module
pybind11_add_module(gap_solver ${GAP_PYTHON_SOURCES})

# Set properties for the Python module
set_target_properties(gap_solver PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Include directories
target_include_directories(gap_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Compile definitions
target_compile_definitions(gap_solver PRIVATE 
    VERSION_INFO="${PROJECT_VERSION}"
)

# Link libraries - core GAP libraries
target_link_libraries(gap_solver PRIVATE
    gap_core
    gap_io
    gap_powerflow_cpu
    gap_admittance_cpu
    gap_lu_cpu
)

# Link CUDA libraries and GPU backends if CUDA is enabled
# NOTE: GPU backend linking currently disabled for Python bindings due to
# CUDA static library linking issues (undefined symbol: fatbinData)
# The C++ GPU backend works perfectly - this only affects Python bindings
if(GAP_ENABLE_CUDA AND FALSE)  # Temporarily disabled
    target_link_libraries(gap_solver PRIVATE
        gap_powerflow_gpu
        gap_admittance_gpu
        gap_lu_gpu
        CUDA::cublas
        CUDA::cusolver
        CUDA::cudart
    )
    
    # Set CUDA properties if needed
    set_property(TARGET gap_solver PROPERTY CUDA_STANDARD 20)
    set_property(TARGET gap_solver PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET gap_solver PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET gap_solver PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Compiler-specific options for the Python module
if(MSVC)
    target_compile_options(gap_solver PRIVATE /W4 /permissive-)
else()
    target_compile_options(gap_solver PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(gap_solver PRIVATE -g -O0)
    else()
        target_compile_options(gap_solver PRIVATE -O3)
    endif()
endif()

# Install the Python module
install(TARGETS gap_solver
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)