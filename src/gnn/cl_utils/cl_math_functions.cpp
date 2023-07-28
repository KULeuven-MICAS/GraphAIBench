#include "cl_math_functions.h"

#ifndef BIN_DIR
#define BIN_DIR "./bin"
#endif

//Possible improvement: global work size can be detected automatically from inputs

void clAvgAggr(size_t vnum, size_t vlen, const cl_mem A_nonzeros, const cl_mem A_idx_ptr, const cl_mem A_nnz_idx, const cl_mem B, cl_mem C, struct oclKernelParamStruct arg_work_groups /*= {NULL, NULL}*/){
    int work_dim = 2;
    std::string kernel_name = "aggr";
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/aggr.pocl";
    struct oclKernelParamStruct work_groups = arg_work_groups;
    
    std::cout << "Loading program avg_aggr: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), kernel_name);
    std::cout << "Program loaded" << std::endl;
    clSetArgs(kernel_name, 0, (void *) &vnum, sizeof(int));
    clSetArgs(kernel_name, 1, (void *) &vlen, sizeof(int));
    clSetArgs(kernel_name, 2, (void *) &A_nonzeros, sizeof(cl_mem));
    clSetArgs(kernel_name, 3, (void *) &A_idx_ptr, sizeof(cl_mem));
    clSetArgs(kernel_name, 4, (void *) &A_nnz_idx, sizeof(cl_mem));
    clSetArgs(kernel_name, 5, (void *) &B, sizeof(cl_mem));
    clSetArgs(kernel_name, 6, (void *) &C, sizeof(cl_mem));
    int work_groups_dim[2] = {vnum, vlen};
    optimizeWorkDimentions(work_dim, work_groups_dim, work_groups);
    clInvokeKernel(kernel_name, work_dim, work_groups.global_work_size, work_groups.local_work_size);
}

void clMatmul(const size_t x, const size_t y, const size_t z, const cl_mem A, const cl_mem B, cl_mem C, struct oclKernelParamStruct arg_work_groups /*= {NULL, NULL}*/, std::string dtype /*= "float"*/){
    int work_dim = 2;
    std::string kernel_name;
    if(dtype == "float") kernel_name = "sgemm";
    else if(dtype == "int") kernel_name = "igemm";
    else throw std::invalid_argument("Invalid data type");

    std::string kernel_path = std::string(BIN_DIR) + "/kernels/" + kernel_name + ".pocl";
    struct oclKernelParamStruct work_groups = arg_work_groups;

    std::cout << "Loading program sgemm: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), kernel_name);
    std::cout << "Program loaded" << std::endl;
	clSetArgs(kernel_name, 0, (void *) &x, sizeof(int));
	clSetArgs(kernel_name, 1, (void *) &y, sizeof(int));
	clSetArgs(kernel_name, 2, (void *) &z, sizeof(int));
	clSetArgs(kernel_name, 3, (void *) &A, sizeof(cl_mem));
	clSetArgs(kernel_name, 4, (void *) &B, sizeof(cl_mem));
	clSetArgs(kernel_name, 5, (void *) &C, sizeof(cl_mem));
    
    std::cout << "Invoking kernel" << std::endl;
    int work_groups_dim[2] = {x, y};
    optimizeWorkDimentions(work_dim, work_groups_dim, work_groups);
    clInvokeKernel(kernel_name, work_dim, &work_groups.global_work_size[0], &work_groups.local_work_size[0]);
}

void clSfilter(const size_t size, const cl_mem src, cl_mem dst, const long long ldc, const float m0, const float m1, const float m2, const float m3, const float m4, const float m5, const float m6, const float m7, const float m8, struct oclKernelParamStruct arg_work_groups /*= {NULL, NULL}*/){
    int work_dim = 2;
    std::string kernel_name = "sfilter";
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/sfilter.pocl";
    struct oclKernelParamStruct work_groups = arg_work_groups;

    std::cout << "Loading program sfilter: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), kernel_name);
    std::cout << "Program loaded" << std::endl;
    clSetArgs(kernel_name, 0, (void *) &size, sizeof(int));
    clSetArgs(kernel_name, 1, (void *) &src, sizeof(cl_mem));
    clSetArgs(kernel_name, 2, (void *) &dst, sizeof(cl_mem));
    clSetArgs(kernel_name, 3, (void *) &ldc, sizeof(long long));
    clSetArgs(kernel_name, 4, (void *) &m0, sizeof(float));
    clSetArgs(kernel_name, 5, (void *) &m1, sizeof(float));
    clSetArgs(kernel_name, 6, (void *) &m2, sizeof(float));
    clSetArgs(kernel_name, 7, (void *) &m3, sizeof(float));
    clSetArgs(kernel_name, 8, (void *) &m4, sizeof(float));
    clSetArgs(kernel_name, 9, (void *) &m5, sizeof(float));
    clSetArgs(kernel_name, 10, (void *) &m6, sizeof(float));
    clSetArgs(kernel_name, 11, (void *) &m7, sizeof(float));
    clSetArgs(kernel_name, 12, (void *) &m8, sizeof(float));
    
    std::cout << "Invoking kernel" << std::endl;
    int work_groups_dim[2] = {size - 2, size - 2};
    optimizeWorkDimentions(work_dim, work_groups_dim, work_groups);
    size_t offset[2] = {1,1};
    clInvokeKernel(kernel_name, work_dim, &work_groups.global_work_size[0], &work_groups.local_work_size[0], &offset[0]);
}

void clSaxpy(const int n, const float a, const cl_mem x, cl_mem y, struct oclKernelParamStruct work_groups /*= {NULL, NULL}*/){
    int work_dim = 1;
    std::string kernel_name = "saxpy";
    std::string kernel_path = std::string(BIN_DIR) + "/kernels/saxpy.pocl";

    std::cout << "Loading program saxpy: " << kernel_path <<std::endl;
    clLoadProgram(kernel_path.c_str(), kernel_name);
    std::cout << "Program loaded" << std::endl;
    clSetArgs(kernel_name, 0, (void *) &n, sizeof(int));
    clSetArgs(kernel_name, 1, (void *) &a, sizeof(float));
    clSetArgs(kernel_name, 2, (void *) &x, sizeof(cl_mem));
    clSetArgs(kernel_name, 3, (void *) &y, sizeof(cl_mem));
    
    std::cout << "Invoking kernel" << std::endl;
    int work_groups_dim[1] = {n};
    optimizeWorkDimentions(work_dim, work_groups_dim, work_groups);
    clInvokeKernel(kernel_name, work_dim, &work_groups.global_work_size[0], &work_groups.local_work_size[0]);
}