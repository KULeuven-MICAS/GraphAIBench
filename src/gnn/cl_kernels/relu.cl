__kernel void relu(int size, __global float *src, __global float *dst)
{
    int i = get_global_id(0);

    if (i >= size) return;
    dst[i] = src[i] > 0 ? src[i] : 0;
}