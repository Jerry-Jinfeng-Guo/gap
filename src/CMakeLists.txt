# Core library (interfaces and abstractions)
add_library(gap_core STATIC
    core/backend_factory.cpp
    logging/logger.cpp
)

if(GAP_ENABLE_CUDA)
    # Add GPU data structures when CUDA is enabled
    target_sources(gap_core PRIVATE
        core/gpu_data_structures.cu
    )
    set_source_files_properties(core/gpu_data_structures.cu PROPERTIES LANGUAGE CUDA)
endif()

target_include_directories(gap_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Enable Position Independent Code for linking into shared libraries
set_target_properties(gap_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA libraries to gap_core if CUDA is enabled
if(GAP_ENABLE_CUDA)
    target_link_libraries(gap_core
        CUDA::cudart
    )
    set_target_properties(gap_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
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
    # Determine which GPU solver to build
    if(NOT DEFINED GAP_GPU_SOLVER_TYPE)
        set(GAP_GPU_SOLVER_TYPE "CUDSS" CACHE STRING "GPU solver type: CUDSS or LU")
    endif()
    
    if(GAP_GPU_SOLVER_TYPE STREQUAL "CUDSS")
        message(STATUS "Building GPU Solver with cuDSS (CUDA Direct Sparse Solver)")
        
        # Find cuDSS (not required - will fall back to LU if not found)
        find_package(cudss QUIET PATHS /usr/local/cudss/lib/cmake/cudss)
        
        if(cudss_FOUND)
            message(STATUS "  cuDSS found at: ${cudss_DIR}")
            
            add_library(gap_lu_gpu SHARED
                solver/lu/gpu/gpu_cudss_solver.cu
            )

            target_include_directories(gap_lu_gpu PUBLIC
                ${CMAKE_SOURCE_DIR}/include
                /usr/local/cudss/include
            )

            target_link_libraries(gap_lu_gpu
                gap_core
                CUDA::cudart
                cudss
            )
        else()
            message(WARNING "cuDSS not found! Falling back to legacy cuSOLVER LU solver.")
            message(WARNING "  To install cuDSS, visit: https://developer.nvidia.com/cudss")
            message(WARNING "  Falling back to: --solver LU")
            set(GAP_GPU_SOLVER_TYPE "LU" CACHE STRING "GPU solver type" FORCE)
            
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
                CUDA::cusparse
            )
        endif()
        
    elseif(GAP_GPU_SOLVER_TYPE STREQUAL "LU")
        message(STATUS "Building GPU Solver with legacy cuSOLVER LU")
        
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
            CUDA::cusparse
        )
        
    else()
        message(FATAL_ERROR "Invalid GAP_GPU_SOLVER_TYPE: ${GAP_GPU_SOLVER_TYPE}. Must be CUDSS or LU")
    endif()

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
        solver/powerflow/gpu/gpu_powerflow_kernels.cu
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
