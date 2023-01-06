#include "global.h"
#include "cl_utils.h"

void clSpmm(size_t x, size_t y, float* A_nonzeros, int* A_idx_ptr, int* A_nnz_idx, const float* B, float* C);
void clMatmul(const size_t x, const size_t y, const size_t z, const float* A, const float*B, float* C); 