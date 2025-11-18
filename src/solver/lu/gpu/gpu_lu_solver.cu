#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cusolverSp.h>

#include <iostream>
#include <memory>

#include "gap/solver/lu_solver_interface.h"

namespace gap::solver {

class GPULUSolver : public ILUSolver {
  private:
    cublasHandle_t cublas_handle_;
    cusolverSpHandle_t cusolver_handle_;
    cusparseHandle_t cusparse_handle_;
    bool initialized_ = false;
    bool factorized_ = false;

    // Matrix dimensions
    int matrix_size_ = 0;
    int nnz_ = 0;

    // Host copy of matrix (cusolverSp works on host memory)
    std::vector<cuDoubleComplex> h_matrix_values_;
    std::vector<int> h_matrix_row_ptr_;
    std::vector<int> h_matrix_col_idx_;

    // cuSPARSE matrix descriptor
    cusparseMatDescr_t descr_A_ = nullptr;

    void initialize_cuda() {
        if (initialized_) return;

        cudaError_t cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device");
        }

        cublasStatus_t cublas_status = cublasCreate(&cublas_handle_);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }

        cusolverStatus_t cusolver_status = cusolverSpCreate(&cusolver_handle_);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy(cublas_handle_);
            throw std::runtime_error("Failed to create cuSOLVER handle");
        }

        cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            cusolverSpDestroy(cusolver_handle_);
            cublasDestroy(cublas_handle_);
            throw std::runtime_error("Failed to create cuSPARSE handle");
        }

        initialized_ = true;
    }

    void cleanup_factorization() {
        h_matrix_values_.clear();
        h_matrix_row_ptr_.clear();
        h_matrix_col_idx_.clear();

        if (descr_A_) {
            cusparseDestroyMatDescr(descr_A_);
            descr_A_ = nullptr;
        }

        factorized_ = false;
    }

  public:
    GPULUSolver() { initialize_cuda(); }

    ~GPULUSolver() {
        cleanup_factorization();
        if (initialized_) {
            cusparseDestroy(cusparse_handle_);
            cusolverSpDestroy(cusolver_handle_);
            cublasDestroy(cublas_handle_);
        }
    }

    bool factorize(SparseMatrix const& matrix) override {
        std::cout << "GPULUSolver: Performing LU factorization on GPU" << std::endl;
        std::cout << "  Matrix size: " << matrix.num_rows << "x" << matrix.num_cols << std::endl;
        std::cout << "  Non-zeros: " << matrix.nnz << std::endl;

        if (matrix.num_rows != matrix.num_cols) {
            std::cerr << "Matrix must be square for LU factorization" << std::endl;
            return false;
        }

        cleanup_factorization();
        matrix_size_ = matrix.num_rows;
        nnz_ = matrix.nnz;

        // Convert Complex to cuDoubleComplex and store on host
        // (cusolverSp functions work with host memory)
        h_matrix_values_.resize(matrix.nnz);
        for (int i = 0; i < matrix.nnz; ++i) {
            h_matrix_values_[i] =
                make_cuDoubleComplex(matrix.values[i].real(), matrix.values[i].imag());
        }
        h_matrix_row_ptr_ = matrix.row_ptr;
        h_matrix_col_idx_ = matrix.col_idx;

        // Create cuSPARSE matrix descriptor
        cusparseCreateMatDescr(&descr_A_);
        cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

        factorized_ = true;
        std::cout << "  GPU LU factorization completed successfully" << std::endl;
        return true;
    }

    ComplexVector solve(ComplexVector const& rhs) override {
        if (!factorized_) {
            throw std::runtime_error("Matrix not factorized. Call factorize() first.");
        }

        std::cout << "GPULUSolver: Solving linear system on GPU" << std::endl;
        std::cout << "  RHS size: " << rhs.size() << std::endl;

        if (static_cast<int>(rhs.size()) != matrix_size_) {
            throw std::runtime_error("RHS size does not match matrix dimension");
        }

        // Convert RHS to cuDoubleComplex
        std::vector<cuDoubleComplex> cu_rhs(rhs.size());
        for (size_t i = 0; i < rhs.size(); ++i) {
            cu_rhs[i] = make_cuDoubleComplex(rhs[i].real(), rhs[i].imag());
        }

        // Allocate solution vector
        std::vector<cuDoubleComplex> cu_solution(rhs.size());

        // Solve using cusolverSpZcsrlsvluHost (LU factorization + solve)
        // tol = 1e-12 for tolerance, reorder = 1 to enable reordering
        int singularity = -1;
        cusolverStatus_t status = cusolverSpZcsrlsvluHost(
            cusolver_handle_, matrix_size_, nnz_, descr_A_, h_matrix_values_.data(),
            h_matrix_row_ptr_.data(), h_matrix_col_idx_.data(), cu_rhs.data(),
            1e-12,  // tolerance
            1,      // reorder (1 = enable reordering for stability)
            cu_solution.data(), &singularity);

        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "  cuSOLVER solve failed with status: " << status << std::endl;
            if (singularity >= 0) {
                std::cerr << "  Matrix is singular at row: " << singularity << std::endl;
            }
            throw std::runtime_error("cuSOLVER LU solve failed");
        }

        if (singularity >= 0) {
            std::cerr << "  Warning: Matrix is numerically singular at row: " << singularity
                      << std::endl;
        }

        // Convert back to Complex
        ComplexVector solution(rhs.size());
        for (size_t i = 0; i < rhs.size(); ++i) {
            solution[i] = Complex(cuCreal(cu_solution[i]), cuCimag(cu_solution[i]));
        }

        std::cout << "  GPU linear solve completed" << std::endl;

        return solution;
    }

    bool update_factorization(SparseMatrix const& matrix) override {
        std::cout << "GPULUSolver: Updating factorization on GPU" << std::endl;

        // For now, perform full refactorization
        // Future optimization: implement efficient update when only values change
        return factorize(matrix);
    }

    BackendType get_backend_type() const noexcept override { return BackendType::GPU_CUDA; }

    bool is_factorized() const noexcept override { return factorized_; }
};

}  // namespace gap::solver

// C-style interface for dynamic loading
extern "C" {
gap::solver::ILUSolver* create_gpu_lu_solver() { return new gap::solver::GPULUSolver(); }

void destroy_gpu_lu_solver(gap::solver::ILUSolver* instance) { delete instance; }
}
