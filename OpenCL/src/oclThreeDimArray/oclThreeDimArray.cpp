#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include "filter_utils.h"

//opencl libries
#include <oclUtils.h>
#include <shrQATest.h>

//#define MAX_GPU_COUNT 8

// Name of the file with the source code for the computation kernel
extern const char* clSourceFile;

#include <cstddef>
#if defined(_WIN32) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

using namespace std;


// Name of the file with the source code for the computation kernel
const char* clSourceFile = "oclThreeDimArray.cl";

// Name of the log file
const char *shrLogFile = "oclThreeDimArray.txt";


void getDevice(const int argc, const char **argv);


int main(int argc, const char **argv)
{
  getDevice(argc, argv); 
  return 0;  
}


void getDevice(const int argc, const char **argv)
{   
	
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
	
    bool ok = true;    
    cl_context        context      = 0;
    cl_platform_id    platform     = 0;
    cl_device_id     *devices      = 0;
    cl_command_queue  commandQueue = 0;
    cl_program        program      = 0;
    cl_kernel         kernel       = 0;
    //cl_event         *kernelEvents = 0;
#ifdef GPU_PROFILING
    cl_ulong          kernelEventStart;
    cl_ulong          kernelEventEnd;
#endif
    double            hostElapsedTimeS;
    char             *cPathAndName = 0;
    char             *cSourceCL = 0;
    size_t            szKernelLength;  
    const size_t 	  globalWorkSize[3] = {outputDepth, outputWidth, outputHeight};
    const size_t 	  localWorkSize[3] = {1, 1, 1};  
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_int            errnum       = 0;
    char              buildOptions[128];
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
   
    // Get the NVIDIA platform
    if (ok)
    {
        shrLog(" oclGetPlatformID...\n");
        errnum = oclGetPlatformID(&platform);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("oclGetPlatformID (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the platform
    if (ok)
    {
        shrLog(" clGetDeviceIDs");
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id) );
        errnum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetDeviceIDs (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Create the OpenCL context
    if (ok)
    {
        shrLog(" clCreateContext...\n");
        context = clCreateContext(0, deviceCount, devices, NULL, NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clCreateContext (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Select target device (device 0 by default)
    if (ok)
    {
        char *device = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "device", &device))
        {
            targetDevice = (cl_uint)atoi(device);
            if (targetDevice >= deviceCount)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -2001, STDERROR);
                shrLog("invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
        {
            free(device);
        }
    }

    // Create a command-queue
    if (ok)
    {
        shrLog(" clCreateCommandQueue\n"); 
        commandQueue = clCreateCommandQueue(context, devices[targetDevice], CL_QUEUE_PROFILING_ENABLE, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clCreateCommandQueue (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Load the kernel from file
    if (ok)
    {
        shrLog(" shrFindFilePath\n"); 
        cPathAndName = shrFindFilePath(clSourceFile, argv[0]);
        if (cPathAndName == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2002, STDERROR);
            shrLog("shrFindFilePath returned null.\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" oclLoadProgSource\n"); 
        cSourceCL = oclLoadProgSource(cPathAndName, "// Preamble\n", &szKernelLength);
        if (cSourceCL == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -2003, STDERROR);
            shrLog("oclLoadProgSource returned null.\n");
            ok = false;
        }
    }

    // Create the program
    if (ok)
    {
        shrLog(" clCreateProgramWithSource\n");
        program = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &szKernelLength, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clCreateProgramWithSource (returned %d).\n", errnum);
            ok = false;
        }
    }

    if (ok)
    {
        shrLog(" clBuildProgram (%s)\n", buildOptions);
        errnum = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            char buildLog[10240];
            clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clBuildProgram (returned %d).\n", errnum);
            shrLog("Log:\n%s\n", buildLog);
            ok = false;
        }
    }

    // Create the kernel
    if (ok)
    {
        shrLog(" clCreateKernel\n");
        //kernel = clCreateKernel(program, "estimate_ekf_parm_1_dim_heston", &errnum);
        kernel = clCreateKernel(program, "ThreeDimArray", &errnum);
        if (kernel == (cl_kernel)NULL || errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clCreateKernel (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the maximum work group size
    size_t maxWorkSize;
    if (ok)
    {
        shrLog(" clGetKernelWorkGroupInfo\n");
        errnum = clGetKernelWorkGroupInfo(kernel, devices[targetDevice], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkSize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clGetKernelWorkGroupInfo (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Set the work group size
    if (ok)
    {
             
        inputCLBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				static_cast<void *>(input1), NULL);
		
	inputCLBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				static_cast<void *>(input2), NULL);
		
	outputCLBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, NULL);
		
	outputCLBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, NULL);		
  
	outputCLBuffer3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				 sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				 NULL, NULL);
		
    }
    
    // Clear result
    if (ok)    
    {
    shrLog("  Clear result with clEnqueueWriteBuffer ...\n");           
    if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("Clear result with clEnqueueWriteBuffer (returned %d).\n", errnum);
            ok = false;     
        }        
     } 
        
      
    // Set the constant arguments
    if (ok)
    {
        shrLog(" clSetKernelArg 0-9\n");                              
        errnum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputCLBuffer1);
		errnum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputCLBuffer2);
		errnum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputCLBuffer1);
		errnum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &outputCLBuffer2);
		errnum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &outputCLBuffer3);
        if (errnum != CL_SUCCESS)
        {
            shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
            shrLog("clSetKernelArg 0-9 (returned %d).\n", errnum);
            ok = false;     
        }
        else shrLog(" clSetKernelArg 0-9 Success\n");
    }
    
      

 // Launch the kernel
	if (ok)
	{
	    shrLog(" clEnqueueNDRangeKernel\n");		                     
	    errnum = clEnqueueNDRangeKernel(
					commandQueue, 
					kernel, 
					3, 
					NULL, 
					globalWorkSize, 
					localWorkSize, 
					0, 
					NULL, 
					NULL);  //globalWorkSize in this sample is const int, not support, should place the address                      
	    if (errnum != CL_SUCCESS)
	     {
			shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
			shrLog("clEnqueueNDRangeKernel (returned %d).\n", errnum);
			ok = false;   
	     }
		else shrLog(" clEnqueueNDRangeKernel Success\n");
	}


	// Read and print the result
	// scalar data types as reference
	// buffer data types as pointers		    
	  
		if (ok)
		{
			shrLog(" clEnqueueReadBuffer\n");	  
			errnum = clEnqueueReadBuffer( commandQueue, outputCLBuffer1, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output1, 0, NULL, NULL);
  
			errnum = clEnqueueReadBuffer( commandQueue, outputCLBuffer2, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output2, 0, NULL, NULL);
			errnum = clEnqueueReadBuffer( commandQueue, outputCLBuffer3, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output3, 0, NULL, NULL);
				if(errnum != CL_SUCCESS) {
					shrLog("Couldn't read the buffer");
					ok = false;   
				} 	
		}
  
  
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

    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (cSourceCL)
        free(cSourceCL);
    if (cPathAndName)
        free(cPathAndName);
    if (commandQueue)
        clReleaseCommandQueue(commandQueue);
    if (devices)
        free(devices);
    if (context)
        clReleaseContext(context);   
}
