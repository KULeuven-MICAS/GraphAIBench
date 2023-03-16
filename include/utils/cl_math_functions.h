#include "global.h"
#include "cl_utils.h"

void clAvgAggr(size_t vnum, size_t vlen, const cl_mem A_nonzeros, const cl_mem A_idx_ptr, const cl_mem A_nnz_idx, const cl_mem B, cl_mem C, struct oclKernelParamStruct work_groups /*= {NULL, NULL}*/);
void clMatmul(struct oclKernelParamStruct work_groups, const size_t x, const size_t y, const size_t z, const cl_mem A, const cl_mem B, cl_mem C); 