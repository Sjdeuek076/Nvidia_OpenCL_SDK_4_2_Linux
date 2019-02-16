//ThreeDimArray.cpp
//Compile with:
//g++ ThreeDimArray.cpp -I/usr/local/cuda-6.0/include/ -L/usr/local/cuda-6.0/lib64/ -L/usr/lib/ -lOpenCL -lstdc++ -o ThreeDimArray

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
//#include <CL/cl.h>

//opencl libries
#include <oclUtils.h>
#include <shrQATest.h>

const unsigned int inputDepth  =     4; //Different dimensions on each axis makes the indices esier to identify.
const unsigned int inputWidth  =     8;
const unsigned int inputHeight =    16;

cl_float input1[inputDepth][inputWidth][inputHeight];
cl_float input2[inputDepth][inputWidth][inputHeight];

const unsigned int outputDepth  =     inputDepth;
const unsigned int outputWidth  =     inputWidth;
const unsigned int outputHeight =     inputHeight;

cl_float output1[outputDepth][outputWidth][outputHeight];
cl_float output2[outputDepth][outputWidth][outputHeight];
cl_float output3[outputDepth][outputWidth][outputHeight];

const char* kernel_source =  
  "__kernel void ThreeDimArray(const __global float * const input1,     \n"
  "			       const __global float * const input2,	\n"
  "			       __global float * const output1,		\n"
  "			       __global float * const output2,		\n"
  "			       __global float * const output3){		\n"								
  " const int x = get_global_id(0);					\n"
  " const int y = get_global_id(1);					\n"
  " const int z = get_global_id(2);					\n"
  " const int max_x = get_global_size(0);				\n"
  " const int max_y = get_global_size(1);				\n"
  " const int max_z = get_global_size(2);				\n"
  " const int idx = x * max_y * max_z + y * max_z + z;			\n"
  " output1[idx] = x;							\n"
  " output2[idx] = y;							\n"
  " output3[idx] = z;							\n"
  " //Uncommnet the next line if you want to see the input values used. \n"
  " //output1[idx] = input1[idx] + input2[idx];				\n"
  "}									\n";

inline void errorCheck(cl_int err, const char * name){
  if(err != CL_SUCCESS){
    std::cerr << "ERROR: " << name << " Error Code: " << err << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CL_CALLBACK contextCallbackFnct(const char * err_info, const void * private_information, size_t tt, void * user_data){
  std::cout << "Error in context:"<< err_info << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv){
  cl_int error_no;
  cl_uint num_of_platforms;
  cl_uint num_of_devices;
  cl_platform_id * platform_ids =0;
  cl_device_id * device_ids =0;
  cl_context context =0;
  cl_command_queue command_queue =0;
  cl_program program =0;
  cl_kernel kernel =0;
  cl_mem inputCLBuffer1;
  cl_mem inputCLBuffer2;
  cl_mem outputCLBuffer1;
  cl_mem outputCLBuffer2;
  cl_mem outputCLBuffer3;
  char buffer[10240];
  
  for (unsigned int i = 0; i < inputDepth; ++i){
    for (unsigned int j = 0; j < inputWidth; ++j){
      for (unsigned int k = 0; k < inputHeight; ++k){
	input1[i][j][k] = 100.0f; //Some values very different from the indices
	input2[i][j][k] = 101.0f;
      }
    }
  }
  
    
  platform_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * num_of_platforms);
  error_no = clGetPlatformID(&platform_ids);
  //error_no = clGetPlatformIDs(num_of_platforms, platform_ids, NULL);
  
  errorCheck((error_no != CL_SUCCESS) ? error_no : (num_of_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
  
  //device_ids = NULL; 
  cl_uint i = 0;
  clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
  printf("  NAME = %s\n", buffer);
  error_no = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, NULL, &num_of_devices);  
  printf("=== %d OpenCL devices: ===\n", num_of_devices);
    
  if (error_no != CL_SUCCESS && error_no != CL_DEVICE_NOT_FOUND){
    errorCheck(error_no, "clGetDeviceIDs");
  }
  else if (num_of_devices > 0){
    device_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * num_of_devices);
    error_no = clGetDeviceIDs( platform_ids, CL_DEVICE_TYPE_GPU, 0, NULL, NULL);
    errorCheck(error_no, "clGetDeviceIDs");
  }
  
  if (device_ids == NULL) {
    std::cout << "No GPU device found" << std::endl;
    exit(-1);
  }
  
  cl_context_properties context_props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[i], 0 };
  context = clCreateContext(context_props, num_of_devices, device_ids, &contextCallbackFnct, NULL, &error_no);
  errorCheck(error_no, "clCreateContext");
  
  size_t kernel_source_length = std::string(kernel_source).length();
  program = clCreateProgramWithSource( context, 1, &kernel_source, &kernel_source_length, &error_no);
  errorCheck(error_no, "clCreateProgramWithSource");

  error_no = clBuildProgram(program, num_of_devices, device_ids, NULL, NULL, NULL);
  errorCheck(error_no, "clBuildProgram");
  clGetProgramBuildInfo(program, *device_ids, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
  fprintf(stderr, "CL Kernel Compilation:\n %s \n", buffer);
  
  kernel = clCreateKernel(program, "ThreeDimArray", &error_no);
  errorCheck(error_no, "clCreateKernel");
    
  inputCLBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				static_cast<void *>(input1), &error_no);
  errorCheck(error_no, "clCreateBuffer(input1)");
  
  inputCLBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				static_cast<void *>(input2), &error_no);
  errorCheck(error_no, "clCreateBuffer(input2)");
  
  outputCLBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(output1)");
  
  outputCLBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(output2)");
  
  outputCLBuffer3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(output3)");
  
  command_queue = clCreateCommandQueue(context, device_ids[0], 0, &error_no);
  errorCheck(error_no, "clCreateCommandCommand_Queue");
  
  error_no  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputCLBuffer1);
  error_no |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputCLBuffer2);
  error_no |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputCLBuffer1);
  error_no |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputCLBuffer2);
  error_no |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &outputCLBuffer3);
  errorCheck(error_no, "clSetKernelArg");
  
  const size_t globalWorkSize[3] = {outputDepth, outputWidth, outputHeight};
  const size_t localWorkSize[3] = {1, 1, 1};
  
  error_no = clEnqueueNDRangeKernel(
				  command_queue,
				  kernel,
				  3,
				  NULL,
				  globalWorkSize,
				  localWorkSize,
				  0,
				  NULL,
				  NULL);
  errorCheck(error_no, "clEnqueueNDRangeKernel");
    
  error_no = clEnqueueReadBuffer( command_queue, outputCLBuffer1, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output1, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - 1");
  error_no = clEnqueueReadBuffer( command_queue, outputCLBuffer2, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output2, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - 2");
  error_no = clEnqueueReadBuffer( command_queue, outputCLBuffer3, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output3, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - 3");
  
  std::ofstream fout("output.txt");
  for (int x = 0; x < outputDepth; x++){
    for (int y = 0; y < outputWidth; y++){
      for (int z = 0; z < outputHeight; z++){
		fout << output1[x][y][z] << " ";
      }	
      fout << std::endl;
    }
    fout << std::endl;
  }
  fout <<  "***********************************\n" << std::endl;
  for (int x = 0; x < outputDepth; x++){
    for (int y = 0; y < outputWidth; y++){
      for (int z = 0; z < outputHeight; z++){
	fout << output2[x][y][z] << " ";
      }	
      fout << std::endl;
    }
    fout << std::endl;
  }
  fout <<  "***********************************\n" << std::endl;
  for (int x = 0; x < outputDepth; x++){
    for (int y = 0; y < outputWidth; y++){
      for (int z = 0; z < outputHeight; z++){
	fout << output3[x][y][z] << " ";
      }	
      fout << std::endl;
    }
    fout << std::endl;
  }
  fout.close();
  return(0);
}
