#pragma once

#include <random>
#include <CL/cl.h>
#include "global.h"
#include <cmath>

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
};

struct oclKernelParamStruct {
  size_t* global_work_size;
  size_t* local_work_size;
};

//TODO init param is useless, should check on validity of h_mem_ptr!
cl_mem clMallocRO(int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
cl_mem clMallocWO(int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
cl_mem clMallocRW(int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
void clFree(cl_mem ob) throw(std::string);
cl_mem clReallocRO(cl_mem ob, int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
cl_mem clReallocWO(cl_mem ob, int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
cl_mem clReallocRW(cl_mem ob, int size, bool init = false, void *h_mem_ptr = NULL) throw(std::string);
void clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(std::string);
void clMemcpyD2H(cl_mem d_mem, int size, void *h_mem) throw(std::string);
template <typename T>
void clInitConstMem(int size, T initValue, cl_mem d_mem_ptr) throw(std::string);
template<typename T>
T random(T range_from, T range_to);
void clInitRangeUniformMem(int size, const float_t a, const float_t b, cl_mem d_mem_ptr) throw(std::string);

void clInit();
void clRelease();
void clSetArgs(int kernel_id, int arg_idx, void *d_mem, int size = 0) throw(std::string);
cl_kernel clLoadProgram(const char* filename, std::string kernel_name);
void clInvokeKernel(int kernel_id, cl_uint work_dim, size_t* g_work_size, size_t* l_work_size) throw(std::string);
void make_global_work_group_even(int work_dim, size_t *&g_work_group, size_t *&l_work_group);