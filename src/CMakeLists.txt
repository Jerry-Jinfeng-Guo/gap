# Core library (interfaces and abstractions)
add_library(gap_core STATIC
    core/backend_factory.cpp
    logging/logger.cpp
)

target_include_directories(gap_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link CUDA libraries to gap_core if CUDA is enabled
if(GAP_ENABLE_CUDA)
    target_link_libraries(gap_core
        CUDA::cudart
    )
endif()

# IO Module
add_library(gap_io SHARED
    io/json_io.cpp
)

target_include_directories(gap_io PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_io
    gap_core
)

# CPU Admittance Matrix Backend
add_library(gap_admittance_cpu SHARED
    admittance/cpu/cpu_admittance_matrix.cpp
)

target_include_directories(gap_admittance_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_admittance_cpu
    gap_core
)

# GPU Admittance Matrix Backend (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    add_library(gap_admittance_gpu SHARED
        admittance/gpu/gpu_admittance_matrix.cu
    )

    target_include_directories(gap_admittance_gpu PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(gap_admittance_gpu
        gap_core
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )

    set_target_properties(gap_admittance_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# CPU LU Solver Backend
add_library(gap_lu_cpu SHARED
    solver/lu/cpu/cpu_lu_solver.cpp
)

target_include_directories(gap_lu_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_lu_cpu
    gap_core
)

# GPU LU Solver Backend (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    add_library(gap_lu_gpu SHARED
        solver/lu/gpu/gpu_lu_solver.cu
    )

    target_include_directories(gap_lu_gpu PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(gap_lu_gpu
        gap_core
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
    )

    set_target_properties(gap_lu_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# CPU Power Flow Solver Backend
add_library(gap_powerflow_cpu SHARED
    solver/powerflow/cpu/cpu_newton_raphson.cpp
)

target_include_directories(gap_powerflow_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_powerflow_cpu
    gap_core
    gap_lu_cpu
)

# GPU Power Flow Solver Backend (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    add_library(gap_powerflow_gpu SHARED
        solver/powerflow/gpu/gpu_newton_raphson.cu
    )

    target_include_directories(gap_powerflow_gpu PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(gap_powerflow_gpu
        gap_core
        gap_lu_gpu
        CUDA::cudart
        CUDA::cublas
    )

    set_target_properties(gap_powerflow_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Main executable
add_executable(gap_main
    main/main.cpp
)

target_include_directories(gap_main PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_main
    gap_core
    gap_io
    ${CMAKE_DL_LIBS}  # For dynamic loading of backends
)

# Set runtime path for shared libraries
set_target_properties(gap_main PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH ON
)
