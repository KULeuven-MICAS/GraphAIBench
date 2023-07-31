__kernel void sgemm (int x, 
                     int y,
                     __global const float *A,
	                   __global const float *B,
	                   __global float *C)
{
  // Thread identifiers
  const int i = get_global_id(0); // Row ID
  const int j = get_global_id(1); // Col ID

  if ((i >= x) || (j >= y)) return;
  C[i * y + j] = A[i * y + j] + B[i];
}