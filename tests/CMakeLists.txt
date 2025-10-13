# Unit tests
add_executable(gap_unit_tests
    unit/unit_test_main.cpp
    unit/test_io.cpp
    unit/test_admittance.cpp
    unit/test_lu_solver.cpp
    unit/test_powerflow.cpp
)

target_include_directories(gap_unit_tests PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_unit_tests
    gap_core
    gap_io
    gap_admittance_cpu
    gap_lu_cpu
    gap_powerflow_cpu
)

# Validation tests
add_executable(gap_validation_tests
    validation/validation_main.cpp
    validation/ieee_test_cases.cpp
)

target_include_directories(gap_validation_tests PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gap_validation_tests
    gap_core
    gap_io
    gap_admittance_cpu
    gap_lu_cpu
    gap_powerflow_cpu
)

# Add tests to CTest
add_test(NAME UnitTests COMMAND gap_unit_tests)
add_test(NAME ValidationTests COMMAND gap_validation_tests)

# GPU tests (only if CUDA is enabled)
if(GAP_ENABLE_CUDA)
    add_executable(gap_gpu_tests
        unit/gpu_test_main.cpp
        unit/test_gpu_admittance.cpp
        unit/test_gpu_lu_solver.cpp
        unit/test_gpu_powerflow.cpp
    )

    target_include_directories(gap_gpu_tests PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(gap_gpu_tests
        gap_core
        gap_admittance_gpu
        gap_lu_gpu
        gap_powerflow_gpu
    )

    add_test(NAME GPUTests COMMAND gap_gpu_tests)
endif()
