//------------------------------------------
//--helper functions for OpenCL
//--programmer:	Jianbin Fang (GTECH)
//--modified: Giuseppe M. Sarda (KULeuven)
//--date:	27/12/2010
//------------------------------------------

#include "cl_utils.h"

struct oclHandleStruct oclHandles;
int DEVICE_ID_INUSED = 0; // deviced id used (default : 0)
cl_uint deviceListSize = 1;

//--create read only buffer for devices
cl_mem clMallocRW(int size, bool init, void *h_mem_ptr) throw(std::string) {
  cl_mem d_mem;
  cl_mem_flags flags = (init) ? CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE;

  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret

 #ifdef ERRMSG
  if (oclHandles.cl_status != CL_SUCCESS)
    throw(std::string("exception in _clMallocRW"));
#endif
  return d_mem;
}
//--create write only buffer for devices
cl_mem clMallocWO(int size, bool init, void *h_mem_ptr) throw(std::string) {
  cl_mem d_mem;
  cl_mem_flags flags = (init) ? CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_WRITE_ONLY;
  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret
#ifdef ERRMSG
  if (oclHandles.cl_status != CL_SUCCESS)
    throw(std::string("exception in _clMallocWO()"));
#endif
  return d_mem;
}
//--create read write buffer for devices
cl_mem clMallocRO(int size, bool init, void *h_mem_ptr) throw(std::string) {
  cl_mem d_mem;
  cl_mem_flags flags = (init) ? CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_ONLY;
  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret
#ifdef ERRMSG
  if (oclHandles.cl_status != CL_SUCCESS){
    std::cout << "cl_status: " << oclHandles.cl_status << std::endl;
    std::cout << "exception in _clMallocRO" << std::endl;
    throw(std::string("exception in _clMallocRO"));
  }
#endif
  return d_mem;
}

void clFree(cl_mem ob) throw(std::string) {
  if (ob != NULL)
    oclHandles.cl_status = clReleaseMemObject(ob);
#ifdef ERRMSG
  oclHandles.error_str = "excpetion in _clFree() ->";
  switch (oclHandles.cl_status) {
  case CL_INVALID_MEM_OBJECT:
    oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
    break;
  case CL_OUT_OF_RESOURCES:
    oclHandles.error_str += "CL_OUT_OF_RESOURCES";
    break;
  case CL_OUT_OF_HOST_MEMORY:
    oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
    break;
  default:
    oclHandles.error_str += "Unkown reseason";
    break;
  }
  if (oclHandles.cl_status != CL_SUCCESS)
    throw(oclHandles.error_str);
#endif
}

void clReallocRO(cl_mem ob, int size, void *h_mem_ptr) throw(std::string) {
    clFree(ob);
#ifdef ERRMSG
  	if (oclHandles.cl_status != CL_SUCCESS)
    	throw(std::string("excpetion in _clReallocateRO()"));
#endif
  	ob = clMallocRO(size, h_mem_ptr);
}
void clReallocWO(cl_mem ob, int size) throw(std::string) {
    clFree(ob);
#ifdef ERRMSG
  	if (oclHandles.cl_status != CL_SUCCESS)
    	throw(std::string("excpetion in _clReallocateWO()"));
#endif
  	ob = clMallocWO(size);
}
void clReallocRW(cl_mem ob, int size, void *h_mem_ptr) throw(std::string) {
    clFree(ob);
#ifdef ERRMSG
  	if (oclHandles.cl_status != CL_SUCCESS)
    	throw(std::string("excpetion in _clReallocateRW()"));
#endif
  	ob = clMallocRW(size, h_mem_ptr);
}

//--transfer data from host to device
void clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(std::string) {
    oclHandles.cl_status = clEnqueueWriteBuffer( oclHandles.queue, d_mem, CL_TRUE, 
                                                 0, size, h_mem_ptr, 0, NULL, NULL);
#ifdef ERRMSG
  if (oclHandles.cl_status != CL_SUCCESS)
    throw(std::string("excpetion in _clMemcpyH2D"));
#endif
}

//--transfer data from device to host
void clMemcpyD2H(cl_mem d_mem, int size, void *h_mem) throw(std::string) {
  	oclHandles.cl_status = clEnqueueReadBuffer(oclHandles.queue, d_mem, CL_TRUE, 0, size, h_mem, 0, 0, 0);
#ifdef ERRMSG
  	oclHandles.error_str = "excpetion in _clCpyMemD2H -> ";
  	switch (oclHandles.cl_status) {
  	case CL_INVALID_COMMAND_QUEUE:
  	  oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
  	  break;
  	case CL_INVALID_CONTEXT:
  	  oclHandles.error_str += "CL_INVALID_CONTEXT";
  	  break;
  	case CL_INVALID_MEM_OBJECT:
  	  oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
  	  break;
  	case CL_INVALID_VALUE:
  	  oclHandles.error_str += "CL_INVALID_VALUE";
  	  break;
  	case CL_INVALID_EVENT_WAIT_LIST:
  	  oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
  	  break;
  	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
  	  oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  	  break;
  	case CL_OUT_OF_HOST_MEMORY:
  	  oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
  	  break;
  	default:
  	  oclHandles.error_str += "Unknown reason";
  	  break;
  	}
  	if (oclHandles.cl_status != CL_SUCCESS)
  	  throw(oclHandles.error_str);
#endif
}

template <typename T>
void clInitConstMem(int size, T initValue, cl_mem d_mem_ptr) throw(std::string) {
    oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue,
                                                d_mem_ptr,
                                                &initValue,
                                                sizeof(T),
                                                0, 
                                                size,
                                                0,
                                                NULL, 
                                                &(oclHandles.event));
    cl_int cl_status = clWaitForEvents(1, &(oclHandles.event));
#ifdef ERRMSG
    if (oclHandles.cl_status != CL_SUCCESS || cl_status != CL_SUCCESS)
        throw(std::string("exception in clFloatInitConstMem"));
#endif
}

template<typename T>
T random(T range_from, T range_to) {
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_real_distribution<T>    distr(range_from, range_to);
    //can be also std::uniform_int_distribution<T>
    return distr(generator);
}
void clInitRangeUniformMem(int size, const float_t a, const float_t b, cl_mem d_mem_ptr) throw(std::string) {
    float_t rn[size];
    for (int i = 0; i < size; i++) {
      	rn[i] = random<float>(a, b);
    }
    clMemcpyH2D(d_mem_ptr, size * sizeof(float_t), rn);
}

int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}

void clInit(){
	std::cout << "Initializing OpenCL objects..." << std::endl;

	cl_platform_id targetPlatform = NULL;

  	oclHandles.context = NULL;
  	oclHandles.devices = NULL;
  	oclHandles.queue = NULL;
  	oclHandles.program = NULL;

	cl_int resultCL;

	//---------------------------------------------------------------------------
	std::cout << "Finding the available platforms and select one..." << std::endl;
	
	cl_uint numPlatforms = 1; // Default: 1 -- only one platform
  cl_platform_id *allPlatforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
  oclHandles.cl_status = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
  
  if (oclHandles.cl_status != CL_SUCCESS) throw(std::string("InitCL()::Error: Getting platform ids (clGetPlatformIDs)"));
  	// Select the target platform. Default: first platform
  targetPlatform = allPlatforms[0];

	//---------------------------------------------------------------------------
	std::cout << "Allocating the device list..." << std::endl;
	
	oclHandles.devices = (cl_device_id *)malloc(deviceListSize * sizeof(cl_device_id));
  	if (oclHandles.devices == 0) throw(std::string("InitCL()::Error: Could not allocate memory."));

  	//-- Next, get the device list data
  	oclHandles.cl_status =clGetDeviceIDs(targetPlatform, CL_DEVICE_TYPE_DEFAULT, deviceListSize,oclHandles.devices, NULL);
  	if (oclHandles.cl_status != CL_SUCCESS) throw(std::string("exception in _clInit -> clGetDeviceIDs-2"));
  
  	oclHandles.context = clCreateContext(NULL, deviceListSize, oclHandles.devices,
                                       		NULL, NULL, &resultCL);
  	if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL)) throw(std::string("InitCL()::Error: Creating Context (clCreateContext)"));

	//---------------------------------------------------------------------------
	std::cout << "Creating the command queue..." << std::endl;

	oclHandles.queue = clCreateCommandQueue(oclHandles.context, oclHandles.devices[DEVICE_ID_INUSED], 0, &resultCL);
  	if ((resultCL != CL_SUCCESS) || (oclHandles.queue == NULL)) {
      std::cerr << "InitCL()::Error: Creating Command Queue. (clCreateCommandQueue)" << std::endl;
      std::cerr << "Error code: " << resultCL << std::endl;
      if (oclHandles.queue == NULL) std::cerr << "Queue is NULL " << std::endl;
      throw(std::string("InitCL()::Creating Command Queue. (clCreateCommandQueue)"));
    }
}

// release CL objects
void clRelease() {
  char errorFlag = false;

  for (int nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++) {
    if (oclHandles.kernel[nKernel] != NULL) {
      cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
      if (resultCL != CL_SUCCESS) {
        std::cerr << "ReleaseCL()::Error: In clReleaseKernel" << std::endl;
        errorFlag = true;
      }
      oclHandles.kernel[nKernel] = NULL;
      printf("clReleaseKernel()\n");
    }
  }

  if (oclHandles.program != NULL) {
    cl_int resultCL = clReleaseProgram(oclHandles.program);
    if (resultCL != CL_SUCCESS) {
      std::cerr << "ReleaseCL()::Error: In clReleaseProgram" << std::endl;
      errorFlag = true;
    }
    oclHandles.program = NULL;
    printf("clReleaseProgram()\n");
  }

  if (oclHandles.queue != NULL) {
    cl_int resultCL = clReleaseCommandQueue(oclHandles.queue);
    if (resultCL != CL_SUCCESS) {
      std::cerr << "ReleaseCL()::Error: In clReleaseCommandQueue" << std::endl;
      errorFlag = true;
    }
    oclHandles.queue = NULL;
    printf("clReleaseCommandQueue()\n");
  }

  if (oclHandles.context != NULL) {
    cl_int resultCL = clReleaseContext(oclHandles.context);
    if (resultCL != CL_SUCCESS) {
      std::cerr << "ReleaseCL()::Error: In clReleaseContext" << std::endl;
      errorFlag = true;
    }
    oclHandles.context = NULL;
    printf("clReleaseContext()\n");
  }

  if (oclHandles.devices != NULL) {
    cl_int resultCL = clReleaseDevice(oclHandles.devices[0]);
    if (resultCL != CL_SUCCESS) {
      std::cerr << "ReleaseCL()::Error: In clReleaseDevice" << std::endl;
      errorFlag = true;
    }
    free(oclHandles.devices);
    printf("clReleaseDevice()\n");
  }

  if (errorFlag)
    throw(std::string("ReleaseCL()::Error encountered."));
}

void clSetArgs(int kernel_id, int arg_idx, void *d_mem, int size /*= 0*/) throw(std::string) {
  //std::cout << "DEBUG::clSetKernelArg() " << std::endl;
  //std::cout << "DEBUG::kernel_id " << kernel_id << std::endl;
  //std::cout << "DEBUG::arg_idx " << arg_idx << std::endl;
  //std::cout << "DEBUG::d_mem " << d_mem << std::endl;
  //std::cout << "DEBUG::size " << size << std::endl;
  if (!size) {
    oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, sizeof(d_mem), d_mem);
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clSetKernelArg() ";
    switch (oclHandles.cl_status) {
    case CL_INVALID_KERNEL:
      oclHandles.error_str += "CL_INVALID_KERNEL";
      break;
    case CL_INVALID_ARG_INDEX:
      oclHandles.error_str += "CL_INVALID_ARG_INDEX";
      break;
    case CL_INVALID_ARG_VALUE:
      oclHandles.error_str += "CL_INVALID_ARG_VALUE";
      break;
    case CL_INVALID_MEM_OBJECT:
      oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
      break;
    case CL_INVALID_SAMPLER:
      oclHandles.error_str += "CL_INVALID_SAMPLER";
      break;
    case CL_INVALID_ARG_SIZE:
      oclHandles.error_str += "CL_INVALID_ARG_SIZE";
      break;
    case CL_OUT_OF_RESOURCES:
      oclHandles.error_str += "CL_OUT_OF_RESOURCES";
      break;
    case CL_OUT_OF_HOST_MEMORY:
      oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
      break;
    default:
      oclHandles.error_str += "Unknown reason";
      break;
    }
    if (oclHandles.cl_status != CL_SUCCESS)
      std::cout << oclHandles.error_str << std::endl;
      throw(oclHandles.error_str);
#endif
  } else {
    oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, size, d_mem);
#ifdef ERRMSG
    oclHandles.error_str = "excpetion in _clSetKernelArg() ";
    switch (oclHandles.cl_status) {
    case CL_INVALID_KERNEL:
      oclHandles.error_str += "CL_INVALID_KERNEL";
      break;
    case CL_INVALID_ARG_INDEX:
      oclHandles.error_str += "CL_INVALID_ARG_INDEX";
      break;
    case CL_INVALID_ARG_VALUE:
      oclHandles.error_str += "CL_INVALID_ARG_VALUE";
      break;
    case CL_INVALID_MEM_OBJECT:
      oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
      break;
    case CL_INVALID_SAMPLER:
      oclHandles.error_str += "CL_INVALID_SAMPLER";
      break;
    case CL_INVALID_ARG_SIZE:
      oclHandles.error_str += "CL_INVALID_ARG_SIZE";
      break;
    case CL_OUT_OF_RESOURCES:
      oclHandles.error_str += "CL_OUT_OF_RESOURCES";
      break;
    case CL_OUT_OF_HOST_MEMORY:
      oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
      break;
    default:
      oclHandles.error_str += "Unknown reason";
      break;
    }
    if (oclHandles.cl_status != CL_SUCCESS){
      std::cout << oclHandles.error_str << "\tcl_status:" << oclHandles.cl_status <<std::endl;
      throw(oclHandles.error_str);
    }
#endif
  }
}

void clLoadProgram(const char* filename, std::string kernel_name) {
	std::cout << "Loading the program..." << std::endl;

	uint8_t* kernel_bin = NULL;
	size_t kernel_size;
	cl_int binary_status = 0, resultCL = 0;
	if (read_kernel_file(filename, &kernel_bin, &kernel_size) != 0) throw(std::string("exception in clLoadProgram -> read_kernel_file"));

	oclHandles.program = clCreateProgramWithBinary(oclHandles.context, 1, &oclHandles.devices[DEVICE_ID_INUSED], &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &resultCL);
	if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)) throw(std::string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));
	free(kernel_bin);

	resultCL = clBuildProgram(oclHandles.program, deviceListSize, oclHandles.devices, NULL, NULL, NULL);
	if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)) {
		std::cerr << "InitCL()::Error: In clBuildProgram" << std::endl;

    	size_t length;
    	resultCL = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED],
    	                                CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
    	if (resultCL != CL_SUCCESS) throw(std::string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

    	char *buffer = (char *)malloc(length);
    	resultCL = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED],
    	    							CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
    	if (resultCL != CL_SUCCESS) throw(std::string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

    	std::cerr << buffer << std::endl;
    	free(buffer);

    	throw(std::string("InitCL()::Error: Building Program (clBuildProgram)"));
  	}

	//Add code here print info about intermediate representation

	//Single kernel launch
	cl_kernel kernel = clCreateKernel(oclHandles.program, kernel_name.c_str(), &resultCL);
    if ((resultCL != CL_SUCCESS) || (kernel == NULL)) {
      	std::string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \"" + kernel_name + "\"";
      	throw(errorMsg);
	}
	oclHandles.kernel.push_back(kernel);
  std::cout << "---Program loaded!" << std::endl;
	//Add code here print allocation info
}

//--enqueue kernel execution
void clInvokeKernel(int kernel_id, cl_uint work_dim, size_t* g_work_size, size_t* l_work_size) throw(std::string) {
  //check on local and global work size is missing here... an improvement could be to add it
  //size_t local_work_size[work_dim] = l_work_size;
  //size_t global_work_size[work_dim] = g_work_size;
	oclHandles.cl_status = clEnqueueNDRangeKernel( oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, g_work_size, l_work_size, 0, 0, NULL);
#ifdef ERRMSG
  oclHandles.error_str = "excpetion in _clInvokeKernel() -> ";
  switch (oclHandles.cl_status) {
  case CL_INVALID_PROGRAM_EXECUTABLE:
    oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
    break;
  case CL_INVALID_COMMAND_QUEUE:
    oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
    break;
  case CL_INVALID_KERNEL:
    oclHandles.error_str += "CL_INVALID_KERNEL";
    break;
  case CL_INVALID_CONTEXT:
    oclHandles.error_str += "CL_INVALID_CONTEXT";
    break;
  case CL_INVALID_KERNEL_ARGS:
    oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
    break;
  case CL_INVALID_WORK_DIMENSION:
    oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
    break;
  case CL_INVALID_GLOBAL_WORK_SIZE:
    oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
    break;
  case CL_OUT_OF_RESOURCES:
    oclHandles.error_str += "CL_OUT_OF_RESOURCES";
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
    break;
  case CL_OUT_OF_HOST_MEMORY:
    oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
    break;
  default:
    oclHandles.error_str += "Unkown reseason";
    break;
  }
  if (oclHandles.cl_status != CL_SUCCESS)
    throw(oclHandles.error_str);
#endif
}