__kernel void aggr (const int vlen,
                    const int vnum,
                    __global const float *A_nnz,
                    __global const int    *A_idx_ptr,
                    __global const int    *A_idx,
                    __global cosnt float *B,
                    __global float *C)
{
    // Thread identifiers
    const int i = get_global_id(0); // Row ID
    const int j = get_global_id(1); // Col ID

    for (auto e = A_idx_ptr[i]; e < A_idx_ptr[i+1]; e++) {
        C[i * y + j] += A[A_idx[e]] * B[i * y + j];
    }
}