__kernel void sgemm (int x, 
                     int y, 
                     int z,
                     __global const float *A,
	                   __global const float *B,
	                   __global float *C)
{
  // Thread identifiers
  const int i = get_global_id(0); // Row ID
  const int j = get_global_id(1); // Col ID

  if (i >= x || j >= y) return;
    // Compute a single element (loop a K)
    float acc = 0.0f;
    for (int k = 0; k < z; k++) {
      acc += A[i*z+k] * B[k*z+j];
    }
    // Store the result
    C[i * y + j] = acc;
}
