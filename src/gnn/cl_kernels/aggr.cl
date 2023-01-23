__kernel void aggr (const int vlen,
                    __global const float *A_nnz,
                    __global const int    *A_idx_ptr,
                    __global const int    *A_idx,
                    __global const float *B,
                    __global float *C)
{
    // Thread identifiers
    const int i = get_global_id(0); // Row ID
    const int j = get_global_id(1); // Col ID

    if (i >= vlen || j >= vlen) return;
    for (int e = A_idx_ptr[i]; e < A_idx_ptr[i+1]; e++) {
        C[i * vlen + j] += A_nnz[A_idx[e]] * B[i * vlen + j];
    }
}