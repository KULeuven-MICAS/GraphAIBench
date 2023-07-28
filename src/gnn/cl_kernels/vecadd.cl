__kernel void saxpy(int size, __global float *src, __global float *dst)
{
    int i = get_global_id(0);

    if (i >= size) return;
    dst[i] += src[i];
}