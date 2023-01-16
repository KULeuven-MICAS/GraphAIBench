#include "cl_utils.h"
#include <unistd.h>
#include <stdlib.h>

static bool almost_equal(float a, float b, int ulp = 21);
static void aggr(const int vlen, const int vnum, const float *A, const int *A_idx_ptr, const int *A_idx, const float *B, float *C);
static int gen_spm(const int vlen, const int vnum, float *A, int *A_idx_ptr, int *A_idx);

int main (int argc, char* argv[]){

    //CPU data 
    int vlen = 16;
    int vnum = 32;
    int nnz;
    float *A;
    float *B = (float*)malloc(vlen*vnum*sizeof(float));
    float *C = (float*)malloc(vlen*vnum*sizeof(float));
    float *H = (float*)malloc(vlen*vnum*sizeof(float));
    int *A_idx_ptr = (int*)malloc((vnum+1)*sizeof(int));
    int *A_idx = (int*)malloc((vnum)*(vlen)*sizeof(int));

    //GPU data
    cl_mem A_d, B_d, C_d, A_idx_ptr_d, A_idx_d;

    //Init data
    for (int i = 0; i < vlen*vnum; i++){
        B[i] = random<float>(-1.0f, 1.0f);
        C[i] = 0.0f;
    }
    nnz = gen_spm(vlen, vnum, A, A_idx_ptr, A_idx);

    //Exec OpenCL
    clInit();
    A_d = clMallocRO(nnz*sizeof(float), true, A);
    B_d = clMallocRO(vlen*vnum*sizeof(float), true, B);
    C_d = clMallocWO(vlen*vnum*sizeof(float), true, C);
    A_idx_ptr_d = clMallocRO((vnum)*sizeof(int), true, A_idx_ptr);
    A_idx_d = clMallocRO(nnz*sizeof(int), true, A_idx);
    clLoadProgram("./bin/kernels/aggr.pocl","aggr");
    clSetArgs(0, 0, (void *) &vlen, sizeof(int));
    clSetArgs(0, 1, (void *) &vnum, sizeof(int));
    clSetArgs(0, 2, (void *) &A_d, sizeof(cl_mem));
    clSetArgs(0, 3, (void *) &A_idx_ptr_d, sizeof(cl_mem));
    clSetArgs(0, 4, (void *) &A_idx_d, sizeof(cl_mem));
    clSetArgs(0, 5, (void *) &B_d, sizeof(cl_mem));
    clSetArgs(0, 6, (void *) &C_d, sizeof(cl_mem));
    size_t g_work_size[2] = {vnum, vlen};
    size_t l_work_size[2] = {16, 16};
    clInvokeKernel(0, 2, g_work_size, l_work_size);
    clMemcpyD2H(C_d, vlen*vnum*sizeof(float), (void*) C);
    clFinish(oclHandle.queue);

    //Exec CPU
    aggr(vlen, vnum, A, A_idx_ptr, A_idx, B, H);

    //Compare
    int err = 0;
    for (int i = 0; i < vlen*vnum; i++){
        if (!almost_equal(C[i], H[i])){
            printf("Error: C[%d] = %f, H[%d] = %f\n", i, C[i], i, H[i]);
            err++;
        }
    }
    if (err == 0){
        printf("Test passed!");
    }else{
        printf("Test failed!");
    }
    clRelease();
    free(A);
    free(B);
    free(C);
    free(H);
    free(A_idx_ptr);
    free(A_idx);
    return 0;
}

static bool almost_equal(float a, float b, int ulp /*= 21*/) {
    union fi_t { int i; float f; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    return std::abs(fa.i - fb.i) <= ulp;
}

//static void aggr(const int vlen, const int vnum, const float *A, const int *A_idx_ptr, const int *A_idx, const float *B, float *C){
//    for (int i = 0; i < vnum; i++){
//        for (int j = A_idx_ptr[i]; j < A_idx_ptr[i+1]; j++){
//            for (int k = 0; k < vlen; k++){
//                C[i*vlen+k] += B[j*vlen+k] * A[A_idx[j]*vlen+k];
//            }
//        }
//    }
//}

static void aggr(const int vlen, const int vnum, const float *A, const int *A_idx_ptr, const int *A_idx, const float *B, float *C){
    vec_t neighbor(vlen);
    for (int i = 0; i < vnum; i++){
        for (int j = A_idx_ptr[i]; j < A_idx_ptr[i+1]; j++){
            for (int k = 0; k < vlen; k++)
                neighbor[k] = B[j*vlen+k] * A[A_idx[j]];
            for (int k = 0; k < vlen; k++)
                C[i*vlen+k] += neighbor[k];
        }
    }
}

static int gen_spm(const int vlen, const int vnum, float *A, int *A_idx_ptr, int *A_idx){
    int nnz = 0;
    int rrr = 0;
    for (int i = 0; i < vnum; i++){
        A_idx_ptr[i] = nnz;
        for (int j = 0; j < vnum; j++){
            rrr = random<int>( -100, 100);
            if (abs(rrr) >= 98){
                A_idx[nnz] = j;
                nnz++;
            }
        }
    }
    A_idx_ptr[vnum] = nnz;
    A = (float*)malloc(nnz*sizeof(float));
    for (int i = 0; i < nnz; i++){
        A[i] = random<float>(nnz, -1.0f, 1.0f);
    }
    return nnz;
}