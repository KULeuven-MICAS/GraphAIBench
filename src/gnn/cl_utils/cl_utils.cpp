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
cl_mem clMallocRW(int size, void *h_mem_ptr) {
  cl_mem d_mem;
  cl_mem_flags flags = (h_mem_ptr!=NULL) ? CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE;
  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret
  oclHandles.error_str = "exception in clMallocRW() ->";
  clErrorHandle(oclHandles.cl_status);
  return d_mem;
}
//--create write only buffer for devices
cl_mem clMallocWO(int size, void *h_mem_ptr){
  cl_mem d_mem;
  cl_mem_flags flags = (h_mem_ptr!=NULL) ? CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_WRITE_ONLY;
  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret
  oclHandles.error_str = "exception in clMallocWO() ->";
  clErrorHandle(oclHandles.cl_status);
  return d_mem;
}
//--create read write buffer for devices
cl_mem clMallocRO(int size, void *h_mem_ptr){
  cl_mem d_mem;
  cl_mem_flags flags = (h_mem_ptr!=NULL) ? CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_ONLY;
  d_mem = clCreateBuffer(oclHandles.context,      //context
                         flags,                   //flags
                         size,                    //size
                         h_mem_ptr,               //*host ptr
                         &oclHandles.cl_status);  //*errcode_ret
  oclHandles.error_str = "exception in clMallocRO() ->";
  clErrorHandle(oclHandles.cl_status);
  return d_mem;
}

void clFree(cl_mem ob){
  if (ob != NULL)
    oclHandles.cl_status = clReleaseMemObject(ob);
  oclHandles.error_str = "exception in clFree() ->";
  clErrorHandle(oclHandles.cl_status);
}

cl_mem clReallocRO(cl_mem ob, int size, void *h_mem_ptr){
  clFree(ob);
  cl_mem new_ob;
  new_ob = clMallocRO(size, h_mem_ptr);
  return new_ob;
}

cl_mem clReallocWO(cl_mem ob, int size, void *h_mem_ptr){
  clFree(ob);
  cl_mem new_ob;
  new_ob = clMallocWO(size, h_mem_ptr);
  return new_ob;
}

cl_mem clReallocRW(cl_mem ob, int size, void *h_mem_ptr){
  clFree(ob);
  cl_mem new_ob;
  new_ob = clMallocRW(size, h_mem_ptr);
  return new_ob;
}

//--transfer data from host to device
void clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr){
  oclHandles.cl_status = clEnqueueWriteBuffer( oclHandles.queue, d_mem, CL_TRUE, 
                                                 0, size, h_mem_ptr, 0, NULL, NULL);
  oclHandles.error_str = "exception in clMemcpyH2D() ->";
  clErrorHandle(oclHandles.cl_status);
}

//--transfer data from device to host
void clMemcpyD2H(cl_mem d_mem, int size, void *h_mem){
  oclHandles.cl_status = clEnqueueReadBuffer(oclHandles.queue, d_mem, CL_TRUE, 0, size, h_mem, 0, 0, 0);
  oclHandles.error_str = "exception in clMemcpyD2H() ->";
  clErrorHandle(oclHandles.cl_status);
}

void clInitRangeUniformMem(int size, const float_t a, const float_t b, cl_mem d_mem_ptr){
    float_t rn[size];
    for (int i = 0; i < size; i++) {
      	rn[i] = random<float>(a, b);
    }
    clMemcpyH2D(d_mem_ptr, size * sizeof(float_t), rn);
    free(rn);
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
	
	cl_uint numPlatforms = deviceListSize;
  cl_platform_id *allPlatforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
  oclHandles.cl_status = clGetPlatformIDs(numPlatforms, allPlatforms, NULL);
  oclHandles.error_str = "exception in clGetPlatformIDs() ->";
  clErrorHandle(oclHandles.cl_status);

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

// get max work group size
size_t clGetMaxWorkGroupSize() {
  size_t max_local_work_size = 0;
  clGetDeviceInfo(oclHandles.devices[DEVICE_ID_INUSED], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_local_work_size, NULL);
  return max_local_work_size;
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

void clSetArgs(std::string kernel_name, int arg_idx, void *d_mem, int size /*= 0*/){
  auto kernel_id = std::find(oclHandles.kernel_ids.begin(), oclHandles.kernel_ids.end(), kernel_name);
  if (kernel_id == oclHandles.kernel_ids.end()) {
    std::cout << "Kernel " << kernel_name << " not loaded." << std::endl;
    throw std::runtime_error("Kernel not loaded.");
  }
  int index = kernel_id -oclHandles.kernel_ids.begin();
  if (!size)
    oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[index], arg_idx, sizeof(d_mem), d_mem);
  else 
    oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[index], arg_idx, size, d_mem);
  oclHandles.error_str = "exception in clSetArgs() ->";
  clErrorHandle(oclHandles.cl_status);
}

void clLoadProgram(const char* filename, std::string kernel_name) {
	auto kernel_id = std::find(oclHandles.kernel_ids.begin(), oclHandles.kernel_ids.end(), kernel_name);
  if (kernel_id != oclHandles.kernel_ids.end()) {
    std::cout << "Kernel " << kernel_name << " already loaded." << std::endl;
    return;
  }
  
  std::cout << "Loading the program: " << kernel_name << std::endl;
	uint8_t* kernel_bin = NULL;
	size_t kernel_size;
	cl_int binary_status = 0, resultCL = 0;
	if (read_kernel_file(filename, &kernel_bin, &kernel_size) != 0) throw(std::string("exception in clLoadProgram -> read_kernel_file"));

	oclHandles.program = clCreateProgramWithBinary(oclHandles.context, 1, &oclHandles.devices[DEVICE_ID_INUSED], &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &resultCL);
	if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)){
    std::cerr << "InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)" << resultCL << std::endl;
    throw(std::string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));
  }
	free(kernel_bin);

	resultCL = clBuildProgram(oclHandles.program, deviceListSize, oclHandles.devices, NULL, NULL, NULL);
	if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL)) {
		std::cerr << "InitCL()::Error: In clBuildProgram" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
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
  oclHandles.kernel_ids.push_back(kernel_name);
  std::cout << "---Program loaded!" << std::endl;
	//Add code here print allocation info
}

//--enqueue kernel execution
void clInvokeKernel(std::string kernel_name, cl_uint work_dim, size_t* g_work_size, size_t* l_work_size){
  auto kernel_id = std::find(oclHandles.kernel_ids.begin(), oclHandles.kernel_ids.end(), kernel_name);
  if (kernel_id == oclHandles.kernel_ids.end()) {
    std::cout << "Kernel " << kernel_name << " not loaded." << std::endl;
    return;
  }
  int index = kernel_id -oclHandles.kernel_ids.begin();
  oclHandles.cl_status = clEnqueueNDRangeKernel( oclHandles.queue, oclHandles.kernel[index] , work_dim, 0, g_work_size, l_work_size, 0, 0, NULL);
  oclHandles.error_str = "exception in clInvokeKernel() ->";
  clErrorHandle(oclHandles.cl_status);
}

int genTotalWorkGroups(int work_dim, size_t* l_work_size){
  int total_work_groups = 1;
  for (int i = 0; i < work_dim; i++) {
    total_work_groups *= l_work_size[i];
  }
  return total_work_groups;
}

void optimizeWorkDimentions(int work_dim, int* work_groups_dim, struct oclKernelParamStruct &work_groups){
  if (work_groups.global_work_size != NULL){
    return;
  }
  work_groups.global_work_size = (size_t*)malloc(work_dim * sizeof(size_t));
  work_groups.local_work_size = (size_t*)malloc(work_dim * sizeof(size_t));
  for (int i = 0; i < work_dim; i++) {
    work_groups.global_work_size[i] = work_groups_dim[i];
    work_groups.local_work_size[i] = 1; //1 will always grant full HW utilization, but not optimal execution time!
  }
  #ifdef VORTEX_RUNTIME
    int hw_virtual_threads_count = NUM_THREADS*NUM_WARPS*NUM_CORES*NUM_CLUSTERS;
    std::cout << "------>[RUNTIME-INFO] NUM_THREADS: " << NUM_THREADS << std::endl;
    std::cout << "------>[RUNTIME-INFO] NUM_CLUSTERS: " << NUM_CLUSTERS << std::endl;
    std::cout << "------>[RUNTIME-INFO] NUM_WARPS: " << NUM_WARPS << std::endl;
    std::cout << "------>[RUNTIME-INFO] NUM_CORES: " << NUM_CORES << std::endl;
    std::cout << "------>[RUNTIME-INFO] HW capabilities: " << hw_virtual_threads_count << std::endl;
    int total_work_items = 1;
    for (int i = 0; i < work_dim; i++) {
      total_work_items *= work_groups.global_work_size[i];
    }
    if (total_work_items < hw_virtual_threads_count) {
      std::cout << "WARNING: total_work_size is smaller than HW capabilities! The execution will be highly inefficient..." << std::endl;
    }
    int local_work_size = 1;
    for (int i = 0; i < work_dim; i++) { //each loop find the optimum for one local work dimention
      if (work_groups.global_work_size[i]<hw_virtual_threads_count)
        std::cout << "WARNING: global_work_size[" << i << "] is smaller than HW capabilities!" << std::endl;
      local_work_size = work_groups.global_work_size[i]/hw_virtual_threads_count;
      work_groups.local_work_size[i] = local_work_size ? local_work_size : 1;
    }
    //avoiding overflows
    size_t max_tot_local_work_size = clGetMaxWorkGroupSize();
    while (genTotalWorkGroups(work_dim, work_groups.local_work_size) > max_tot_local_work_size) {
      for (int i = work_dim-1; i >= 0; i--) {
        if (work_groups.local_work_size[i] > 1) {
          work_groups.local_work_size[i] /= 2;
          break;
        }
      }
    }
  #endif
    make_global_work_group_even(work_dim, work_groups.global_work_size, work_groups.local_work_size);
    for (int i = 0; i < work_dim; i++) {
      std::cout << "------>[RUNTIME-INFO] global_work_size[" << i << "] = " << work_groups.global_work_size[i] << std::endl;
      std::cout << "------>[RUNTIME-INFO] local_work_size[" << i << "] = " << work_groups.local_work_size[i] << std::endl;
    }
  return;
}


void make_global_work_group_even(int work_dim, size_t *&g_work_group, size_t *&l_work_group){
  for (int i = 0; i < work_dim; i++) 
    if (g_work_group[i] % l_work_group[i] != 0) 
      g_work_group[i] = g_work_group[i] + (l_work_group[i] - (g_work_group[i] % l_work_group[i]));
}

void clErrorHandle(cl_int status){
  switch (status) {
  case CL_SUCCESS :
    oclHandles.error_str.clear();
    return;
    break;
  case CL_DEVICE_NOT_FOUND :
    oclHandles.error_str += "CL_DEVICE_NOT_FOUND";
    break;
  case CL_DEVICE_NOT_AVAILABLE :
    oclHandles.error_str += "CL_DEVICE_NOT_AVAILABLE";
    break;
  case CL_COMPILER_NOT_AVAILABLE :
    oclHandles.error_str += "CL_COMPILER_NOT_AVAILABLE";
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE :
    oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    break;
  case CL_OUT_OF_RESOURCES :
    oclHandles.error_str += "CL_OUT_OF_RESOURCES";
    break;
  case CL_OUT_OF_HOST_MEMORY :
    oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE :
    oclHandles.error_str += "CL_PROFILING_INFO_NOT_AVAILABLE";
    break;
  case CL_MEM_COPY_OVERLAP :
    oclHandles.error_str += "CL_MEM_COPY_OVERLAP";
    break;
  case CL_IMAGE_FORMAT_MISMATCH :
    oclHandles.error_str += "CL_IMAGE_FORMAT_MISMATCH";
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED :
    oclHandles.error_str += "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    break;
  case CL_BUILD_PROGRAM_FAILURE :
    oclHandles.error_str += "CL_BUILD_PROGRAM_FAILURE";
    break;
  case CL_MAP_FAILURE :
    oclHandles.error_str += "CL_MAP_FAILURE";
    break;
  case CL_MISALIGNED_SUB_BUFFER_OFFSET :
    oclHandles.error_str += "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    break;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
    oclHandles.error_str += "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    break;
  case CL_COMPILE_PROGRAM_FAILURE :
    oclHandles.error_str += "CL_COMPILE_PROGRAM_FAILURE";
    break;
  case CL_LINKER_NOT_AVAILABLE :
    oclHandles.error_str += "CL_LINKER_NOT_AVAILABLE";
    break;
  case CL_LINK_PROGRAM_FAILURE :
    oclHandles.error_str += "CL_LINK_PROGRAM_FAILURE";
    break;
  case CL_DEVICE_PARTITION_FAILED :
    oclHandles.error_str += "CL_DEVICE_PARTITION_FAILED";
    break;
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE :
    oclHandles.error_str += "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    break;
  case CL_INVALID_VALUE :
    oclHandles.error_str += "CL_INVALID_VALUE";
    break;
  case CL_INVALID_DEVICE_TYPE :
    oclHandles.error_str += "CL_INVALID_DEVICE_TYPE";
    break;
  case CL_INVALID_PLATFORM :
    oclHandles.error_str += "CL_INVALID_PLATFORM";
    break;
  case CL_INVALID_DEVICE :
    oclHandles.error_str += "CL_INVALID_DEVICE";
    break;
  case CL_INVALID_CONTEXT :
    oclHandles.error_str += "CL_INVALID_CONTEXT";
    break;
  case CL_INVALID_QUEUE_PROPERTIES :
    oclHandles.error_str += "CL_INVALID_QUEUE_PROPERTIES";
    break;
  case CL_INVALID_COMMAND_QUEUE :
    oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
    break;
  case CL_INVALID_HOST_PTR :
    oclHandles.error_str += "CL_INVALID_HOST_PTR";
    break;
  case CL_INVALID_MEM_OBJECT :
    oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
    oclHandles.error_str += "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    break;
  case CL_INVALID_IMAGE_SIZE :
    oclHandles.error_str += "CL_INVALID_IMAGE_SIZE";
    break;
  case CL_INVALID_SAMPLER :
    oclHandles.error_str += "CL_INVALID_SAMPLER";
    break;
  case CL_INVALID_BINARY :
    oclHandles.error_str += "CL_INVALID_BINARY";
    break;
  case CL_INVALID_BUILD_OPTIONS :
    oclHandles.error_str += "CL_INVALID_BUILD_OPTIONS";
    break;
  case CL_INVALID_PROGRAM :
    oclHandles.error_str += "CL_INVALID_PROGRAM";
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE :
    oclHandles.error_str += "CL_INVALID_PROGRAM_EXECUTABLE";
    break;
  case CL_INVALID_KERNEL_NAME :
    oclHandles.error_str += "CL_INVALID_KERNEL_NAME";
    break;
  case CL_INVALID_KERNEL_DEFINITION :
    oclHandles.error_str += "CL_INVALID_KERNEL_DEFINITION";
    break;
  case CL_INVALID_KERNEL :
    oclHandles.error_str += "CL_INVALID_KERNEL";
    break;
  case CL_INVALID_ARG_INDEX :
    oclHandles.error_str += "CL_INVALID_ARG_INDEX";
    break;
  case CL_INVALID_ARG_VALUE :
    oclHandles.error_str += "CL_INVALID_ARG_VALUE";
    break;
  case CL_INVALID_ARG_SIZE :
    oclHandles.error_str += "CL_INVALID_ARG_SIZE";
    break;
  case CL_INVALID_KERNEL_ARGS :
    oclHandles.error_str += "CL_INVALID_KERNEL_ARGS";
    break;
  case CL_INVALID_WORK_DIMENSION :
    oclHandles.error_str += "CL_INVALID_WORK_DIMENSION";
    break;
  case CL_INVALID_WORK_GROUP_SIZE :
    oclHandles.error_str += "CL_INVALID_WORK_GROUP_SIZE";
    break;
  case CL_INVALID_WORK_ITEM_SIZE :
    oclHandles.error_str += "CL_INVALID_WORK_ITEM_SIZE";
    break;
  case CL_INVALID_GLOBAL_OFFSET :
    oclHandles.error_str += "CL_INVALID_GLOBAL_OFFSET";
    break;
  case CL_INVALID_EVENT_WAIT_LIST :
    oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
    break;
  case CL_INVALID_EVENT :
    oclHandles.error_str += "CL_INVALID_EVENT";
    break;
  case CL_INVALID_OPERATION :
    oclHandles.error_str += "CL_INVALID_OPERATION";
    break;
  case CL_INVALID_GL_OBJECT :
    oclHandles.error_str += "CL_INVALID_GL_OBJECT";
    break;
  case CL_INVALID_BUFFER_SIZE :
    oclHandles.error_str += "CL_INVALID_BUFFER_SIZE";
    break;
  case CL_INVALID_MIP_LEVEL :
    oclHandles.error_str += "CL_INVALID_MIP_LEVEL";
    break;
  case CL_INVALID_GLOBAL_WORK_SIZE :
    oclHandles.error_str += "CL_INVALID_GLOBAL_WORK_SIZE";
    break;
  case CL_INVALID_PROPERTY :
    oclHandles.error_str += "CL_INVALID_PROPERTY";
    break;
  case CL_INVALID_IMAGE_DESCRIPTOR :
    oclHandles.error_str += "CL_INVALID_IMAGE_DESCRIPTOR";
    break;
  case CL_INVALID_COMPILER_OPTIONS :
    oclHandles.error_str += "CL_INVALID_COMPILER_OPTIONS";
    break;
  case CL_INVALID_LINKER_OPTIONS :
    oclHandles.error_str += "CL_INVALID_LINKER_OPTIONS";
    break;
  case CL_INVALID_DEVICE_PARTITION_COUNT :
    oclHandles.error_str += "CL_INVALID_DEVICE_PARTITION_COUNT";
    break;
  default:
    oclHandles.error_str += "Unkown reseason";
    break;
  }
  oclHandles.error_str += "cl_status: " + std::to_string(status);
  std::cerr << oclHandles.error_str << std::endl;
  throw std::runtime_error(oclHandles.error_str);
}