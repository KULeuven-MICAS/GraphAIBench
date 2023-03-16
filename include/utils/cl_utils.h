#pragma once

#include <random>
#include <CL/cl.h>
#include "global.h"
#include <assert.h>

#include <cmath>
#include <stdexcept>
#include <exception>
#include <algorithm>

#ifdef VORTEX
#include "VX_config.h"
#endif

extern struct oclHandleStruct oclHandles;
struct oclHandleStruct {
  cl_context context;
  cl_device_id *devices;
  cl_command_queue queue;
  cl_program program;
  cl_int cl_status;
  cl_event event;
  std::string error_str;
  std::vector<cl_kernel> kernel;
  std::vector<std::string> kernel_ids;
};

struct oclKernelParamStruct {
  size_t* global_work_size;
  size_t* local_work_size;
};

cl_mem clMallocRO(int size, void *h_mem_ptr = NULL);
cl_mem clMallocWO(int size, void *h_mem_ptr = NULL);
cl_mem clMallocRW(int size, void *h_mem_ptr = NULL);
void clFree(cl_mem ob);
cl_mem clReallocRO(cl_mem ob, int size, void *h_mem_ptr = NULL);
cl_mem clReallocWO(cl_mem ob, int size, void *h_mem_ptr = NULL);
cl_mem clReallocRW(cl_mem ob, int size, void *h_mem_ptr = NULL);

void clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr);
void clMemcpyD2H(cl_mem d_mem, int size, void *h_mem);

template<typename T>
void clInitConstMem(int size, T initValue, cl_mem d_mem_ptr);

template<typename T>
T random(T range_from, T range_to);
void clInitRangeUniformMem(int size, const float_t a, const float_t b, cl_mem d_mem_ptr);

void clInit();
void clRelease();
void clSetArgs(std::string kernel_name, int arg_idx, void *d_mem, int size = 0);
void clLoadProgram(const char* filename, std::string kernel_name);
void clInvokeKernel(std::string kernel_name, cl_uint work_dim, size_t* g_work_size, size_t* l_work_size);

void make_global_work_group_even(int work_dim, size_t *&g_work_group, size_t *&l_work_group);
void optimizeWorkDimentions(int work_dim, int* work_groups_dim, struct oclKernelParamStruct &work_groups);