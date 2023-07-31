#include "global.h"
#include "cl_utils.h"

void clAvgAggr(size_t vnum, size_t vlen, const cl_mem A_nonzeros, const cl_mem A_idx_ptr, const cl_mem A_nnz_idx, const cl_mem B, cl_mem C, struct oclKernelParamStruct work_groups = {NULL, NULL});
void clMatmul(const size_t x, const size_t y, const size_t z, const cl_mem A, const cl_mem B, cl_mem C, struct oclKernelParamStruct work_groups = {NULL, NULL}, std::string dtype = "float");
void clSfilter(const size_t size, const cl_mem src, cl_mem dst, const long long ldc, const float m0, const float m1, const float m2, const float m3, const float m4, const float m5, const float m6, const float m7, const float m8, struct oclKernelParamStruct arg_work_groups = {NULL, NULL});
void clSaxpy(const int n, const float a, const cl_mem x, cl_mem y, struct oclKernelParamStruct work_groups = {NULL, NULL});
void clVecadd(const int n, const cl_mem x, cl_mem y, struct oclKernelParamStruct work_groups = {NULL, NULL});
void clNearestNeighbor(const int numRecords, const cl_mem locations, cl_mem distances, const float lat, const float lng, struct oclKernelParamStruct work_groups = {NULL, NULL});
void clRelu(const int n, const cl_mem x, cl_mem y, struct oclKernelParamStruct work_groups = {NULL, NULL});
void clBiasAdd(const int x, const int y, const cl_mem A, const cl_mem B, cl_mem C, struct oclKernelParamStruct work_groups = {NULL, NULL});