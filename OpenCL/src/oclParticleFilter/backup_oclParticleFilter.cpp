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

// The values are set to give reasonable performance, they can be changed
// but note that setting an excessively large work group size can result in
// build failure due to insufficient local memory, even though it would be
// clamped before execution. This is because the maximum work group size
// cannot be determined before the build.
/*#define k_localWorkX    32
#define k_localWorkY    8
#define k_localWorkMin  128
#define k_localWorkMax  1024



#ifndef _OCLFDTD3D_H_
#define _OCLFDTD3D_H_

// The values are set to give reasonable runtimes, they can
// be changed but note that running very large dimensions can
// take a very long time and you should avoid running on your
// primary display in this case.
#define k_dim_min           96
#define k_dim_max           376
#define k_dim_qa            248

// Note that the maximum radius is defined here as 4 since the
// minimum work group height is 4, if you have a larger work
// group then you can increase the radius accordingly.
#define k_radius_min        2
#define k_radius_max        4
#define k_radius_default    4

// The values are set to give reasonable runtimes, they can
// be changed but note that running a very large number of
// timesteps can take a very long time and you should avoid
// running on your primary display in this case.
#define k_timesteps_min     1
#define k_timesteps_max     10
#define k_timesteps_default 5

#endif*/


using namespace std;


// Name of the file with the source code for the computation kernel
const char* clSourceFile = "oclParticleFilter.cl";

// Name of the log file
const char *shrLogFile = "oclParticleFilter.txt";


void getDevice(const int argc, const char **argv);
void read_lines(string& fname, vector<float>& out);

//global values; h stand for host
float *h_log_stock_prices;
float *h_u;
float *h_v;
float *h_estimates;
float h_omega =0.01;
float h_theta=0.02;
float h_xi=0.03;
float h_rho=0.04;
float h_muS=0.05;
int   h_n_stock_prices;
 
ofstream output_file;




void read_lines(string& fname, vector<float>& out) {
	std::ifstream fhandle(fname.c_str());
	char line[1000];
	int index = 0;
	do {
		fhandle.getline(line, 1000);
		if(!fhandle.eof())
			out.push_back(atof(line));

	} while(!fhandle.eof());

}


int main(int argc, const char **argv)
{

      // create device buffers 
  vector<float> prices;
  string input_file_name = "Final-Project.csv";	 
  read_lines(input_file_name, prices);
  cout<<"Found "<<prices.size()<<" prices"<<endl;
  h_n_stock_prices = prices.size();
  cout << "n_stock_prices: "<<h_n_stock_prices<<endl;
 
  h_log_stock_prices = new float[h_n_stock_prices];
  h_u = new float[h_n_stock_prices]; 
  h_v = new float[h_n_stock_prices];
  h_estimates = new float[h_n_stock_prices + 1];

  for(int i = 0; i < prices.size(); i++) {
		h_log_stock_prices[i] = log(prices[i]);
  }
	
  getDevice(argc, argv);
 
  return 0;  
}


void getDevice(const int argc, const char **argv)
{
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
    //size_t            globalWorkSize[2]; // Total # of work items in the 1D range
    //size_t            localWorkSize[2];  // # of work items in the 1D work group        
    size_t            globalWorkSize; // 
    size_t            localWorkSize;  // 
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_int            errnum       = 0;
    char              buildOptions[128];
    cl_mem input_log_stock_prices, output_u, output_v, output_estimates;         
   
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
        kernel = clCreateKernel(program, "estimate_ekf_parm_1_dim_heston", &errnum);
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
		// one dimension -- vector
        localWorkSize = 256;
        globalWorkSize = shrRoundUp((int)localWorkSize, h_n_stock_prices);
        shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           globalWorkSize, localWorkSize, (globalWorkSize % localWorkSize + globalWorkSize/localWorkSize)); 
        
        // two dimension -- matrix
        //userWorkSize = CLAMP(userWorkSize, k_localWorkMin, maxWorkSize);
        //localWorkSize[0] = k_localWorkX;
        //localWorkSize[1] = userWorkSize / k_localWorkX;
        //globalWorkSize[0] = localWorkSize[0] * (unsigned int)ceil((float)dimx / localWorkSize[0]);
        //globalWorkSize[1] = localWorkSize[1] * (unsigned int)ceil((float)dimy / localWorkSize[1]);
        //shrLog(" set local work group size to %dx%d\n", localWorkSize[0], localWorkSize[1]);
        //shrLog(" set total work size to %dx%d\n", globalWorkSize[0], globalWorkSize[1]);
        //shrLog(" set total work size to %dx%d\n", globalWorkSize);       
    
    // Create buffers      	                         
	    input_log_stock_prices = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * globalWorkSize, h_log_stock_prices, NULL);   
	    output_u = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * globalWorkSize, NULL, NULL);							 							 
	    output_v = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, NULL, NULL);							     
	    output_estimates = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, NULL, NULL);							     
    }
    
    // Clear result
    if (ok)    
    {
    shrLog("  Clear result with clEnqueueWriteBuffer ...\n");     
    errnum  = clEnqueueWriteBuffer(commandQueue, output_u, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize, h_u, 0, NULL, NULL);
    errnum  = clEnqueueWriteBuffer(commandQueue, output_v, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize, h_v, 0, NULL, NULL);
    errnum  = clEnqueueWriteBuffer(commandQueue, output_estimates, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize, h_estimates, 0, NULL, NULL);
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
        errnum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_log_stock_prices);
        errnum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_u);
        errnum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_v);        
        errnum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_estimates);
        errnum |= clSetKernelArg(kernel, 4, sizeof(float), &h_omega);
        errnum |= clSetKernelArg(kernel, 5, sizeof(float), &h_theta);
        errnum |= clSetKernelArg(kernel, 6, sizeof(float), &h_xi);
        errnum |= clSetKernelArg(kernel, 7, sizeof(float), &h_rho);
        errnum |= clSetKernelArg(kernel, 8, sizeof(float), &h_muS);
        errnum |= clSetKernelArg(kernel, 9, sizeof(int),   &h_n_stock_prices);        
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
		//errnum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);                       
		errnum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);  //globalWorkSize in this sample is const int, not support, should place the address                      
		if (errnum != CL_SUCCESS)
		{
			shrLogEx(LOGBOTH | ERRORMSG, errnum, STDERROR);
			shrLog("clEnqueueNDRangeKernel (returned %d).\n", errnum);
			ok = false;   
		}
		else shrLog(" clEnqueueNDRangeKernel Success\n");
	}


	// Read and print the result
	if (ok)
	{
	    shrLog(" clEnqueueReadBuffer\n");
	    errnum = clEnqueueReadBuffer(commandQueue, output_u, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, h_u, 0, NULL, NULL);
	    errnum = clEnqueueReadBuffer(commandQueue, output_v, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, h_v, 0, NULL, NULL);		    		    
	    errnum = clEnqueueReadBuffer(commandQueue, output_estimates, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, h_estimates, 0, NULL, NULL);		    		    
		if(errnum != CL_SUCCESS) {
			  shrLog("Couldn't read the buffer");
			  ok = false;   
		} 	
	}	
	  
	  
	  
	  output_file.open("myout.txt");
	  output_file<<"no,log_stock_price,u,v,estimates"<<endl;	  
	  for(unsigned int i=0; i<h_n_stock_prices; i+=1) {         
         output_file<<i<<","<<h_log_stock_prices[i]<<","<<h_u[i]<<","<<h_v[i]<<","<<h_estimates[i]<<endl;
         cout<<" num: "<<i<<" lnprice: "<<h_log_stock_prices[i]<<" u:  "<<h_u[i]<<" v: "<<h_v[i]<<" estimates: "<<h_estimates[i]<<endl;         
      }

   
        
    // Cleanup
    /*if (kernelEvents)
    {
        for (int it = 0 ; it < timesteps ; it++)
        {
            if (kernelEvents[it])
                clReleaseEvent(kernelEvents[it]);
        }
        free(kernelEvents);
    }*/
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
   clReleaseMemObject(input_log_stock_prices);
   clReleaseMemObject(output_u);
   clReleaseMemObject(output_v);
   clReleaseMemObject(output_estimates);
   free(h_u);
   free(h_v);
   free(h_estimates);
}



