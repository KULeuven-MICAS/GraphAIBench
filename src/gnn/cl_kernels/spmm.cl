__kernel void spmm (const int x,
                    const int y, 
                    const int z,
                    __global float *A_nnz,
                    __global int *A_idx_ptr,
                    __global int *A_idx,
                    __global float *B,
                    __global float *C)
{
    // Thread identifiers
    const int i = get_global_id(0); // Row ID
    const int j = get_global_id(1); // Col ID
    for (auto e = A_idx_ptr[i]; e < A_idx_ptr[i+1]; e++) {
        C[i * y + j] += A[A_idx[e]] * B[i * y + j];
    }
}

/*  CUDA IMPLEMENTATION with 32 threads per warp

__global__ void spmv_csr_vector_kernel (    const int num_rows ,
                                            const int * ptr ,
                                            const int * indices ,
                                            const float * data ,
                                            const float * x ,
                                            float * y)
{
    __shared__ float vals [];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
    int warp_id = thread_id / 32; // global warp index
    int lane = thread_id & (32 - 1); // thread index within the warp
    // one warp per row
    int row = warp_id ;
    if ( row < num_rows ){
        int row_start = ptr [ row ];
        int row_end = ptr [ row +1];
        // compute running sum per thread
        vals [ threadIdx.x ] = 0;
        for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
        vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];
        // parallel reduction in shared memory
        if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
        if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
        if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
        if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
        if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
        // first thread writes the result
        if ( lane == 0)
        y[ row ] += vals [ threadIdx.x ];
    }
}

*/