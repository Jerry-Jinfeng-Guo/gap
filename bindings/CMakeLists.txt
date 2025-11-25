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

# Enable CUDA support when available
# LTO (Link Time Optimization) conflict is resolved by disabling LTO below
if(GAP_ENABLE_CUDA)
    list(APPEND GAP_PYTHON_SOURCES 
        cuda_bindings.cpp
    )
    message(STATUS "Python bindings: CUDA support enabled")
else()
    message(STATUS "Python bindings: CPU-only")
endif()

# Disable LTO when CUDA is enabled to prevent symbol conflicts
# Must be set BEFORE calling pybind11_add_module
if(GAP_ENABLE_CUDA)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
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

# Disable LTO (Link Time Optimization) when CUDA is enabled
# LTO from the host compiler conflicts with CUDA's device code linking
# causing "undefined symbol: fatbinData" errors
if(GAP_ENABLE_CUDA)
    set_target_properties(gap_solver PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
    # Force disable LTO in both compile and link stages
    target_compile_options(gap_solver PRIVATE -fno-lto)
    target_link_options(gap_solver PRIVATE -fno-lto)
endif()

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

# Link GPU libraries when CUDA is enabled
if(GAP_ENABLE_CUDA)
    target_link_libraries(gap_solver PRIVATE
        gap_powerflow_gpu
        gap_admittance_gpu
        gap_lu_gpu
    )
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