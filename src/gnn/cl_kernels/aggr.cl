__kernel void aggr (int vnum,
                    int vlen,
                    __global const float  *A,
                    __global const int    *A_idx_ptr,
                    __global const int    *A_idx,
                    __global const float  *B,
                    __global float        *C)
{
    // Thread identifiers
    const int i = get_global_id(0); // Row ID // Vertex ID
    const int j = get_global_id(1); // Col ID // Feature ID

    if ((i >= vnum) || (j >= vlen)) return; 
    for (int e = A_idx_ptr[i]; e < A_idx_ptr[i+1]; e++) {
        int dst = A_idx[e];
        C[i * vlen + j] += A[e] * B[dst * vlen + j];
    }
}