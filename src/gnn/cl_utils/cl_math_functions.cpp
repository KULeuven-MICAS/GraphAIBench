#include "cl_math_functions.h"

#ifndef BIN_DIR
#define BIN_DIR "./bin"
#endif

//Possible improvement: global work size can be detected automatically from inputs

void clAvgAggr(struct oclKernelParamStruct work_groups, size_t vnum, size_t vlen, const cl_mem A_nonzeros, const cl_mem A_idx_ptr, const cl_mem A_nnz_idx, const cl_mem B, cl_mem C){
    int work_dim = 2;
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/aggr.pocl";
    std::cout << "Loading program avg_aggr: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), "aggr");
    std::cout << "Program loaded" << std::endl;
    clSetArgs(1, 0, (void *) &vlen, sizeof(int));
    clSetArgs(1, 1, (void *) &A_nonzeros, sizeof(cl_mem));
    clSetArgs(1, 2, (void *) &A_idx_ptr, sizeof(cl_mem));
    clSetArgs(1, 3, (void *) &A_nnz_idx, sizeof(cl_mem));
    clSetArgs(1, 4, (void *) &B, sizeof(cl_mem));
    clSetArgs(1, 5, (void *) &C, sizeof(cl_mem));
    make_global_work_group_even(work_dim, work_groups.global_work_size, work_groups.local_work_size);
    clInvokeKernel(0, 2, work_groups.global_work_size, work_groups.local_work_size);
}

void clMatmul(struct oclKernelParamStruct work_groups, const size_t x, const size_t y, const size_t z, const cl_mem A, const cl_mem B, cl_mem C){
    int work_dim = 2;
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/sgemm.pocl";
    std::cout << "Loading program sgemm: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), "sgemm");
    std::cout << "Program loaded" << std::endl;
	clSetArgs(0, 0, (void *) &x, sizeof(int));
	clSetArgs(0, 1, (void *) &y, sizeof(int));
	clSetArgs(0, 2, (void *) &z, sizeof(int));
	clSetArgs(0, 3, (void *) &A, sizeof(cl_mem));
	clSetArgs(0, 4, (void *) &B, sizeof(cl_mem));
	clSetArgs(0, 5, (void *) &C, sizeof(cl_mem));
    std::cout << "Invoking kernel" << std::endl;
    make_global_work_group_even(work_dim, work_groups.global_work_size, work_groups.local_work_size);
    clInvokeKernel(0, work_dim, &work_groups.global_work_size[0], &work_groups.local_work_size[0]);
}