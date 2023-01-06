#include "cl_utils.h"
#include <unistd.h>

#define MAX_THREADS_PER_BLOCK 256

static bool almost_equal(float a, float b, int ulp = 21);
static void parse_args(int argc, char **argv);
static void matmul(float *C, const float* A, const float *B, int M, int N, int K);

int size = 32;
int argv_block_size = MAX_THREADS_PER_BLOCK;

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

	size_t nbytes = size * size * sizeof(float);
  cl_mem a_d = NULL, b_d = NULL, c_d = NULL;
	float *a_h = NULL, *b_h = NULL, *c_h = NULL; 
	int width = size;

  std::cout << "Create OpenCL context..." << std::endl;
  clInit();
  
  //----------------------------------------------------------
  std::cout << "Testing extra functionalities" << std::endl;

  float *a, *b, *c; 
  b = (float*) malloc(sizeof(float));
  c = (float*) malloc(sizeof(float));
  b[0] = 1.0f;
  c[0] = 0.0f;
  //a = (float*) clMallocRW(sizeof(float),true,(void*) b);
  a = (float*) clMallocRW(sizeof(float));
  std::cout << "Malloc OKK" << std::endl;
  clMemcpyH2D(cl_mem(a), sizeof(float), (void*) b);
  //clMemcpyD2H(cl_mem(a), sizeof(float), (void*) c);
  std::cout << "B:" << b[0] <<std::endl;
  std::cout << "C:" << c[0] <<std::endl;
  clFree(cl_mem(a));
  free(b);
  free(c);
  //----------------------------------------------------------

  std::cout << "OpenCL context created" << std::endl;
  a_h = (float *) malloc(nbytes);
	b_h = (float *) malloc(nbytes);
	c_h = (float *) malloc(nbytes);
	for (int i = 0; i < size; i++) {
		a_h[i] = random<float>(-1.0f, 1.0f);
		b_h[i] = random<float>(-1.0f, 1.0f);
		c_h[i] = 0.2f;
  }
  a_d = clMallocRO(nbytes, true, a_h);
  b_d = clMallocRO(nbytes, true, b_h);
  c_d = clMallocRW(nbytes, true, c_h);
  clLoadProgram("./bin/kernels/sgemm.pocl","sgemm");
  std::cout << "Program loaded" << std::endl;
	clSetArgs(0, 0, (void *) &a_d, sizeof(cl_mem));
  std::cout << "A param set" << std::endl; 
	clSetArgs(0, 1, (void *) &b_d, sizeof(cl_mem));
  std::cout << "B param set" << std::endl; 
	clSetArgs(0, 2, (void *) &c_d, sizeof(cl_mem));
  std::cout << "C param set" << std::endl; 
	clSetArgs(0, 3, (void *) &width, sizeof(width));
  std::cout << "N param set" << std::endl; 
	clMemcpyH2D(a_d, nbytes, (void *) a_h);
	clMemcpyH2D(b_d, nbytes, (void *) b_h);
  size_t g_work_size[2] = {size, size};
  size_t l_work_size[2] = {argv_block_size, argv_block_size};
	clInvokeKernel(0, 2, g_work_size, l_work_size);
	clFinish(oclHandles.queue);
	clMemcpyD2H(c_d, nbytes, (void *) c_h);

	std::cout << "Checking the results..." << std::endl;

	int errors = 0;
  	float* h_ref = (float*)malloc(nbytes);
  	matmul(h_ref, a_h , b_h, size, size, size);
  	for (int i = 0; i < (size * size); i++) {
    	if (!almost_equal(c_h[i], h_ref[i])) {
      		if (errors < 100) 
        		printf("*** error: [%d] expected=%f, actual=%f\n", i, h_ref[i], c_h[i]);
      		++errors;
    	}
  	}  
  	free(h_ref);
  	if (errors != 0) {
   		printf("FAILED! - %d errors\n", errors);    
  	} else {
    	printf("PASSED!\n");
  	}

	clRelease();
	free(a_h);
	free(b_h);
	free(c_h);
	clFree(a_d);
	clFree(b_d);
	clFree(c_d);

	return 0;
}

static bool almost_equal(float a, float b, int ulp /*= 21*/) {
    union fi_t { int i; float f; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    return std::abs(fa.i - fb.i) <= ulp;
}

static void show_usage() {
  printf("Usage: [-n size] [-b block size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "b:n:h?")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'b':
      argv_block_size = atoi(optarg);
      if (argv_block_size > MAX_THREADS_PER_BLOCK)
        fprintf(stderr, "\tERROR: block_size=%d, MAX=%d\n", argv_block_size, MAX_THREADS_PER_BLOCK);
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  if (size < 2) {
    printf("Error: invalid size!\n");
    exit(-1);
  }

  printf("Workload size=%d\n", size);
  printf("Block size=%d\n", argv_block_size);
}

static void matmul(float *C, const float* A, const float *B, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
          acc += A[k * M + m] * B[n * K + k];
      }
      C[n * M + m] = acc;
    }
  }
}