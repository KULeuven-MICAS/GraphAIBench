#pragma once

#include <CL/cl.h>
#include "global.h"

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

cl_mem clMallocRO(int size, void *h_mem_ptr) throw(std::string);
cl_mem clMallocWO(int size) throw(std::string);
cl_mem clMallocRW(int size, void *h_mem_ptr) throw(std::string);
void clFree(cl_mem ob) throw(std::string);
void clReallocRO(cl_mem ob, int size, void *h_mem_ptr) throw(std::string);
void clReallocWO(cl_mem ob, int size) throw(std::string);
void clReallocRW(cl_mem ob, int size, void *h_mem_ptr) throw(std::string);
void clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(std::string);
void clMemcpyD2H(cl_mem d_mem, int size, void *h_mem) throw(std::string);
template <typename T>
void clInitConstMem(int size, T initValue, cl_mem d_mem_ptr) throw(std::string);
void clInitRangeUniformMem(int size, const float_t a, const float_t b, cl_mem d_mem_ptr) throw(std::string);

void clInit();
void clRelease();
void clSetArgs(int kernel_id, int arg_idx, void *d_mem, int size = 0) throw(std::string);