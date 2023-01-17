#include "cl_math_functions.h"

#ifndef BIN_DIR
#define BIN_DIR "./bin"
#endif


void clAvgAggr(size_t vnum, size_t vlen, float* A_nonzeros, int* A_idx_ptr, int* A_nnz_idx, const float* B, float* C){
    std::cout << "Not implemented yet" << std::endl;
    throw("Too bad!");
}

void clMatmul(const struct oclKernelParam work_groups, const size_t x, const size_t y, const size_t z, const float* A, const float* B, float* C){
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/sgemm.pocl";
    std::cout << "Loading program sgemm: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path,"sgemm");
    std::cout << "Program loaded" << std::endl;
	clSetArgs(0, 0, (void *) &x, sizeof(size_t));
	clSetArgs(0, 1, (void *) &y, sizeof(size_t));
	clSetArgs(0, 2, (void *) &z, sizeof(size_t));
	clSetArgs(0, 3, (void *) &A, sizeof(float*));
	clSetArgs(0, 4, (void *) &B, sizeof(float*));
	clSetArgs(0, 5, (void *) &C, sizeof(float*));
    clInvokeKernel(0, 2, work_groups.global_work_size, work_groups.local_work_size);
}