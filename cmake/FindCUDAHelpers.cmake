# Find CUDA helpers
find_package(CUDAToolkit)

# Helper function to find CUDA compute capability
function(cuda_detect_installed_gpus out_variable)
    if(NOT CUDA_gpu_detect_output)
        set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu)

        file(WRITE ${cufile} ""
            "#include <cstdio>\n"
            "int main() {\n"
            "    int count = 0;\n"
            "    if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
            "    if (count == 0) return -1;\n"
            "    for (int device = 0; device < count; ++device) {\n"
            "        cudaDeviceProp prop;\n"
            "        if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
            "            std::printf(\"%d.%d \", prop.major, prop.minor);\n"
            "    }\n"
            "    return 0;\n"
            "}\n")

        execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "-std=c++14" "--run" "${cufile}"
                        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                        RESULT_VARIABLE cuda_return_code
                        OUTPUT_VARIABLE cuda_output
                        ERROR_VARIABLE cuda_error_output
                        OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(cuda_return_code EQUAL 0)
            set(CUDA_gpu_detect_output ${cuda_output} CACHE INTERNAL "Detected CUDA compute architectures")
        endif()
    endif()

    if(CUDA_gpu_detect_output)
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()

# Macro to add CUDA compile flags
macro(cuda_add_compile_options target)
    if(CMAKE_CUDA_COMPILER)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )
        
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        )
    endif()
endmacro()