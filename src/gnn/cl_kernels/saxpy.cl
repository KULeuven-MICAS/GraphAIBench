__kernel void saxpy(int size, float factor, __global float *src, __global float *dst)
{
    int i = get_global_id(0);

    if (i >= size) return;
    dst[i] += src[i] * factor;
}