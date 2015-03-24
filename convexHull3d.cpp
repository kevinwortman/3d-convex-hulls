/*
 * A bottom-up adaptation of the minimalist divide-and-conquer 
 * algorithm for 3D convex hulls using OpenCL.
 *
 * File: convexHull3d.cpp
 * OpenCL File Used: convexHull3d.cl
 *
 * Last Modified: June 12, 2012
 * 
 * Jeffrey M. White
 * Kevin A. Wortman
 */

#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <math.h>

// GLOBALS 
#define SDK_SUCCESS 0
#define SDK_FAILURE 1
const int NIL = -1;
const float INF = 1e30;

struct Point { 
    cl_float x;
	cl_float y;
	cl_float z;  
    cl_int prev;
	cl_int next;
};

cl_uint n;		// number of points

// Host memory
Point  *P;		// list of all points
cl_int *A;		// Main list of events
cl_int *B;		// Temp list of events
cl_int *M;		// m value during merges

// The memory buffer that is used as input/output for OpenCL kernel
cl_mem pointBuffer;
cl_mem listBufferA;
cl_mem listBufferB;
cl_mem mBuffer;

cl_context context;
cl_device_id *devices;
cl_command_queue commandQueue;
cl_program program;

// This program uses only one kernel and this serves as a handle to it
cl_kernel  kernel;

// FUNCTION DECLARATIONS
int initializeHost(void);
int readInPoints(void);
int initializeCL(void);
std::string convertToString(const char * filename);
int runCLKernels(void);
void printHull();
void act(int point);
int cleanupCL(void);
void cleanupHost(void);

timeval startTime;
timeval endTime;
bool flag;

int main(int argc, char * argv[])
{
    // Initialize Host application 
    if(initializeHost() != SDK_SUCCESS)
        return SDK_FAILURE;

    // Initialize OpenCL resources
    if(initializeCL() != SDK_SUCCESS)
        return SDK_FAILURE;

    // Run the CL program
    if(runCLKernels() != SDK_SUCCESS)
        return SDK_FAILURE;

    // Print Faces on Hull
    //printHull();

    // Releases OpenCL resources 
    if(cleanupCL()!= SDK_SUCCESS)
        return SDK_FAILURE;

    // compute millisecond time difference
    long elapsed_mtime;    // elapsed time in milliseconds
    long elapsed_seconds;  // diff between seconds counter
    long elapsed_useconds; // diff between microseconds counter
    elapsed_seconds  = endTime.tv_sec  - startTime.tv_sec;
    elapsed_useconds = endTime.tv_usec - startTime.tv_usec;
    elapsed_mtime = ((elapsed_seconds) * 1000 + elapsed_useconds/1000.0) + 0.5;

    printf("\nComputed the 3d Convex Hull of %d Points in  %ld milliseconds\n", n, elapsed_mtime);

    // Release host resources
    cleanupHost();

    return SDK_SUCCESS;
}

/*
 * Initialize host resources
 */
int initializeHost(void)
{
	// get number of points
	std::cin >> n;

    // Allocate and initialize memory used by host
	cl_uint pointSizeInBytes = n * sizeof(Point);
	P = (Point *) malloc(pointSizeInBytes);
	if(!P)
	{
		std::cout << "Error: Failed to allocate Points P memory on host\n";
        return SDK_FAILURE;
    }

	cl_uint eventListSizeInBytes = 2* n * sizeof(cl_uint);
    A = (cl_int *) malloc(eventListSizeInBytes);
    if(!A)
    {
        std::cout << "Error: Failed to allocate event list A memory on host\n";
        return SDK_FAILURE;
    }

	B = (cl_int *) malloc(eventListSizeInBytes);
    if(!B)
    {
        std::cout << "Error: Failed to allocate event list B memory on host\n";
        return SDK_FAILURE;
    }

    cl_uint mSizeInBytes = sizeof(cl_uint);
    M = (cl_int *) malloc(mSizeInBytes);
    if(!M)
    {
        std::cout << "Error: Failed to allocate m value memory on host\n";
        return SDK_FAILURE;
    }

	// Read In Points 
    if(readInPoints() != SDK_SUCCESS)
        return SDK_FAILURE;

    return SDK_SUCCESS;
}

/*
 * Read in a sorted (by x-coordinate) list of general points into P for n points
 */
int readInPoints(void)
{
	// read in points into point array
    for (int i=0; i< n; i++) 
	{
		std::cin >> P[i].x; 
		std::cin >> P[i].y; 
		std::cin >> P[i].z;
		P[i].prev = P[i].next = NIL;
    }

	return SDK_SUCCESS;
}

/*
 * Converts the contents of a file into a string
 */
std::string convertToString(const char *filename)
{
    size_t size;
    char*  str;
    std::string s;

    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            std::cout << "Memory allocation failed";
            return NULL;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
    
        s = str;
        delete[] str;
        return s;
    }
    else
    {
        std::cout << "\nFile containg the kernel code(\".cl\") not found. Please copy the required file in the folder containg the executable.\n";
        exit(1);
    }
    return NULL;
}

/*
 * OpenCL initialization:
 * Create Context, device list, command queue, OpenCL memory buffer objects, load CL file, compile CL code, link CL source, build program and kernel objects
 */
int initializeCL(void)
{
    cl_int status = 0;
    size_t deviceListSize;

    // Get Platform
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: Getting Platforms. (clGetPlatformsIDs)\n";
        return SDK_FAILURE;
    }

	std::cout << "Number of platforms found:\t" << numPlatforms << std::endl;
    
    if(numPlatforms > 0)
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if(status != CL_SUCCESS)
        {
            std::cout << "Error: Getting Platform Ids. (clGetPlatformsIDs)\n";
            return SDK_FAILURE;
        }
        for(unsigned int i=0; i < numPlatforms; ++i)
        {
            char pbuff[100];
            status = clGetPlatformInfo(
                        platforms[i],
                        CL_PLATFORM_VENDOR,
                        sizeof(pbuff),
                        pbuff,
                        NULL);
            if(status != CL_SUCCESS)
            {
                std::cout << "Error: Getting Platform Info.(clGetPlatformInfo)\n";
                return SDK_FAILURE;
            }
			
			std::cout << "Platform " << i << ":\t\t\t" << pbuff << std::endl;

            platform = platforms[i];
            if(!strcmp(pbuff, "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
        delete platforms;
    }

    if(NULL == platform)
    {
        std::cout << "NULL platform found so Exiting Application." << std::endl;
        return SDK_FAILURE;
    }

	std::cout << "Platform ID selected:\t\t" << platform << std::endl;	

    // Create context using the platform selected.
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

    context = clCreateContextFromType(cps, 
                                      CL_DEVICE_TYPE_CPU, 
                                      NULL, 
                                      NULL, 
                                      &status);
    if(status != CL_SUCCESS) 
    {  
        std::cout << "Error: Creating Context. (clCreateContextFromType)\n";
        return SDK_FAILURE; 
    }

    // get size of device list data
    status = clGetContextInfo(context, 
                              CL_CONTEXT_DEVICES, 
                              0, 
                              NULL, 
                              &deviceListSize);
    if(status != CL_SUCCESS) 
    {  
        std::cout <<
            "Error: Getting Context Info \
            (device list size, clGetContextInfo)\n";
        return SDK_FAILURE;
    }
	
	std::cout << "Device List Size:\t\t" << deviceListSize << std::endl;

    devices = (cl_device_id *)malloc(deviceListSize);
    if(devices == 0)
    {
        std::cout << "Error: No devices found.\n";
        return SDK_FAILURE;
    }

    // get device list data
    status = clGetContextInfo(
                 context, 
                 CL_CONTEXT_DEVICES, 
                 deviceListSize, 
                 devices, 
                 NULL);
    if(status != CL_SUCCESS) 
    { 
        std::cout <<
            "Error: Getting Context Info \
            (device list, clGetContextInfo)\n";
        return SDK_FAILURE;
    }

    // Create command queue for single device
    commandQueue = clCreateCommandQueue(
                       context, 
                       devices[0], 
                       0, 
                       &status);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Creating Command Queue. (clCreateCommandQueue)\n";
        return SDK_FAILURE;
    }

    // Create cl_buffer objects from host buffer
    pointBuffer = clCreateBuffer(
                      context, 
                      CL_MEM_READ_WRITE,
                      sizeof(Point) * n,
                      NULL, 
                      &status);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: clCreateBuffer (pointBuffer)\n" << status << std::endl;
        return SDK_FAILURE;
    }

    listBufferA = clCreateBuffer(
                       context, 
                       CL_MEM_READ_WRITE,
                       sizeof(cl_int) * 2 * n,
                       NULL, 
                       &status);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: clCreateBuffer (listBufferA)\n";
        return SDK_FAILURE;
    }

	listBufferB = clCreateBuffer(
                       context, 
                       CL_MEM_READ_WRITE,
                       sizeof(cl_int) * 2 * n,
                       NULL, 
                       &status);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: clCreateBuffer (listBufferB)\n";
        return SDK_FAILURE;
    }

	mBuffer = clCreateBuffer(
                       context, 
                       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                       sizeof(cl_int),
                       M, 
                       &status);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: clCreateBuffer (mBuffer)\n";
        return SDK_FAILURE;
    }

    // Build Kernel
    const char * filename  = "convexHull3d.cl";
    std::string  sourceStr = convertToString(filename);
    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };

    program = clCreateProgramWithSource(
                  context, 
                  1, 
                  &source,
                  sourceSize,
                  &status);
    if(status != CL_SUCCESS) 
    { 
      	std::cout << "Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n";
      	return SDK_FAILURE;
    }

    // create a cl program executable for all the devices specified
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: Building Program (clBuildProgram)\n";
        return SDK_FAILURE; 
    }

    // get a kernel object handle for a kernel with the given name
    kernel = clCreateKernel(program, "convexHull3dKernel", &status);
    if(status != CL_SUCCESS) 
    {  
        std::cout << "Error: Creating Kernel from program. (clCreateKernel)\n";
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


/*
 * Run OpenCL program:
 * Bind host variables to kernel arguments 
 * Run the CL kernel
 */
int runCLKernels(void)
{
    cl_int status;
    cl_uint maxDims;
    cl_event events[2];
    size_t globalThreads[1];
    size_t localThreads[1];
    size_t maxWorkGroupSize;
    size_t maxWorkItemSizes[3];

    // Analyze proper workgroup size for the kernel.
    // Query device capabilities
    status = clGetDeviceInfo(
        devices[0], 
        CL_DEVICE_MAX_WORK_GROUP_SIZE, 
        sizeof(size_t), 
        (void*)&maxWorkGroupSize, 
        NULL);
    if(status != CL_SUCCESS) 
    {  
        std::cout << "Error: Getting Device Info. (clGetDeviceInfo)\n";
        return SDK_FAILURE;
    }
	std::cout << "Max Work Group Size:\t\t" << maxWorkGroupSize << std::endl;

    status = clGetDeviceInfo(
        devices[0], 
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
        sizeof(cl_uint), 
        (void*)&maxDims, 
        NULL);
    if(status != CL_SUCCESS) 
    {  
        std::cout << "Error: Getting Device Info. (clGetDeviceInfo)\n";
        return SDK_FAILURE;
    }
	std::cout << "Max Work Item Dimensions:\t" << maxDims << std::endl;

    status = clGetDeviceInfo(
        devices[0], 
        CL_DEVICE_MAX_WORK_ITEM_SIZES, 
        sizeof(size_t)*maxDims,
        (void*)maxWorkItemSizes,
        NULL);
    if(status != CL_SUCCESS) 
    {  
        std::cout << "Error: Getting Device Info. (clGetDeviceInfo)\n";
        return SDK_FAILURE;
    }
	std::cout << "Max Work Item Sizes:\t\t" << maxWorkItemSizes[0] << std::endl;

    globalThreads[0] = n;
    localThreads[0]  = 1;

    if(localThreads[0] > maxWorkGroupSize ||
        localThreads[0] > maxWorkItemSizes[0])
    {
        std::cout << "Unsupported: Device does not support requested number of work items.";
        return SDK_FAILURE;
    }

    // Set appropriate arguments to the kernel
    
    // the point list to the kernel
    status = clSetKernelArg(
                    kernel, 
                    0, 
                    sizeof(cl_mem), 
                    (void *)&pointBuffer);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: Setting kernel argument. (point list)\n";
        return SDK_FAILURE;
    }

    // the event list A to the kernel
    status = clSetKernelArg(
                    kernel, 
                    1, 
                    sizeof(cl_mem), 
                    (void *)&listBufferA);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: Setting kernel argument. (event list A)\n";
        return SDK_FAILURE;
    }

	// the event list B to the kernel
    status = clSetKernelArg(
                    kernel, 
                    2, 
                    sizeof(cl_mem), 
                    (void *)&listBufferB);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: Setting kernel argument. (event list B)\n";
        return SDK_FAILURE;
    }

    // the m value to the kernel
    status = clSetKernelArg(
                    kernel, 
                    3, 
                    sizeof(cl_mem), 
                    (void *)&mBuffer);
    if(status != CL_SUCCESS) 
    { 
        std::cout << "Error: Setting kernel argument. (m value)\n";
        return SDK_FAILURE;
    }

	// write Points to buffer
	status = clEnqueueWriteBuffer(commandQueue, pointBuffer, CL_TRUE, 0, sizeof(Point) * n, P, 0, NULL, NULL);
	if(status != CL_SUCCESS) 
	{ 
	    std::cout << "Error: clEnqueueWriteBuffer(pointBuffer)\n";
	    return SDK_FAILURE;
	}

	// write event list A to buffer
	status = clEnqueueWriteBuffer(commandQueue, listBufferA, CL_TRUE, 0, sizeof(cl_int) * 2 * n, A, 0, NULL, NULL);
	if(status != CL_SUCCESS) 
	{ 
	    std::cout << "Error: clEnqueueWriteBuffer(listBufferA)\n";
	    return SDK_FAILURE;
	}

	// write event list B to buffer
	status = clEnqueueWriteBuffer(commandQueue, listBufferB, CL_TRUE, 0, sizeof(cl_int) * 2 * n, B, 0, NULL, NULL);
	if(status != CL_SUCCESS) 
	{ 
	    std::cout << "Error: clEnqueueWriteBuffer(listBufferB)\n";
	    return SDK_FAILURE;
	}

	int merges;
	M[0] = 2;
	int k = n/2;
	bool swap = true;

	std::cout << std::endl;
	std::cout << "Begin the main merge process..." << std::endl;

	// get start time
    gettimeofday(&startTime, NULL);

	// MAIN MERGE ROUTINE
	do {
	    globalThreads[0] = merges = k;
		if(status != CL_SUCCESS) 
		{ 
		    std::cout << "Error: clEnqueueWriteBuffer((m value)\n";
		    return SDK_FAILURE;
		}

		// enqueue a kernel run call 
		status = clEnqueueNDRangeKernel(
		             commandQueue,
		             kernel,
		             1,
		             NULL,
		             globalThreads,
		             localThreads,
		             0,
		             NULL,
		             &events[0]);
		if(status != CL_SUCCESS) 
		{ 
		    std::cout <<
		        "Error: Enqueueing kernel onto command queue. \
		        (clEnqueueNDRangeKernel)\n";
		    return SDK_FAILURE;
		}

		// wait for the kernel call to finish execution
		status = clWaitForEvents(1, &events[0]);
		if(status != CL_SUCCESS) 
		{ 
		    std::cout <<
		        "Error: Waiting for kernel run to finish. \
		        (clWaitForEvents)\n";
		    return SDK_FAILURE;
		}

		status = clReleaseEvent(events[0]);
		if(status != CL_SUCCESS) 
		{ 
		    std::cout <<
		        "Error: Release event object. \
		        (clReleaseEvent)\n";
		    return SDK_FAILURE;
		}

		if (swap) {
	        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &listBufferA);
	        status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &listBufferB);
			if(status != CL_SUCCESS) 
			{ 
		    	std::cout << "Error: clSetKernelArg\n";
		    	return SDK_FAILURE;
			}	
			swap = !swap;
	    }
	    else {
	        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &listBufferA);
			status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &listBufferB);
			if(status != CL_SUCCESS) 
			{ 
		    	std::cout << "Error: clSetKernelArg\n";
		    	return SDK_FAILURE;
			}
			swap = !swap;
	    }

	    // update m value
	    M[0]*=2;
		
	    // update k
	    k = k/2;
	} while(merges > 1);

	// get end time
    gettimeofday(&endTime, NULL);

	flag = swap;

    // enqueue readBuffer to read the output back
    status = clEnqueueReadBuffer(
                commandQueue,
                pointBuffer,
                CL_TRUE,
                0,
                n * sizeof(Point),
                P,
                0,
                NULL,
                &events[1]);
    
    if(status != CL_SUCCESS) 
    { 
        std::cout << 
            "Error: clEnqueueReadBuffer failed. \
             (clEnqueueReadBuffer)\n";
        return SDK_FAILURE;
    }
    
    // Wait for the read buffer to finish execution
    status = clWaitForEvents(1, &events[1]);
    if(status != CL_SUCCESS) 
    { 
        std::cout <<
            "Error: Waiting for read buffer call to finish. \
            (clWaitForEvents)\n";
        return SDK_FAILURE;
    }

	status = clEnqueueReadBuffer(
                commandQueue,
                listBufferA,
                CL_TRUE,
                0,
                2 * n * sizeof(cl_int),
                A,
                0,
                NULL,
                &events[1]);
    
    if(status != CL_SUCCESS) 
    { 
        std::cout << 
            "Error: clEnqueueReadBuffer failed. \
             (clEnqueueReadBuffer)\n";
        return SDK_FAILURE;
    }
    
    // Wait for the read buffer to finish execution
    status = clWaitForEvents(1, &events[1]);
    if(status != CL_SUCCESS) 
    { 
        std::cout <<
            "Error: Waiting for read buffer call to finish. \
            (clWaitForEvents)\n";
        return SDK_FAILURE;
    }

	status = clEnqueueReadBuffer(
                commandQueue,
                listBufferB,
                CL_TRUE,
                0,
                2 * n * sizeof(cl_int),
                B,
                0,
                NULL,
                &events[1]);
    
    if(status != CL_SUCCESS) 
    { 
        std::cout << 
            "Error: clEnqueueReadBuffer failed. \
             (clEnqueueReadBuffer)\n";
        return SDK_FAILURE;
    }
    
    // Wait for the read buffer to finish execution
    status = clWaitForEvents(1, &events[1]);
    if(status != CL_SUCCESS) 
    { 
        std::cout <<
            "Error: Waiting for read buffer call to finish. \
            (clWaitForEvents)\n";
        return SDK_FAILURE;
    }    

    status = clReleaseEvent(events[1]);
    if(status != CL_SUCCESS) 
    { 
        std::cout <<
            "Error: Release event object. \
            (clReleaseEvent)\n";
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

/*
 * Print faces on the hull
 */
void printHull()
{
	std::cout << "\nFaces on the hull:\n" << std::endl;
	if(flag) {
		for (int i = 0; B[i] != NIL; i++) { 
				std::cout << "Face " << i << ":     " << P[ B[i] ].prev << " " << B[i] << " " << P[ B[i] ].next << "\n";
				act( B[i] );
		}
	}
	else {
		for (int i = 0; A[i] != NIL; i++) { 
				std::cout << "Face " << i << ":     " << P[ A[i] ].prev << " " << A[i] << " " << P[ A[i] ].next << "\n";
				act( A[i] );
		}
	}
	std::cout << std::endl;
}

void act(int point) {
	if ( P[ P[point].prev ].next != point) {   // insert
		P[ P[point].prev ].next = P[ P[point].next ].prev = point;
	}
	else { // delete
		P[ P[point].prev ].next = P[point].next;
		P[ P[point].next ].prev = P[point].prev;
	}
}

/*
 * Release OpenCL resources (Context, Memory etc.) 
 */
int cleanupCL(void)
{
    cl_int status;

    // Clean up the opencl resources used
    
    status = clReleaseKernel(kernel);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseKernel \n";
        return SDK_FAILURE; 
    }
    status = clReleaseProgram(program);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseProgram\n";
        return SDK_FAILURE; 
    }
    status = clReleaseMemObject(pointBuffer);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseMemObject (pointBuffer)\n";
        return SDK_FAILURE; 
    }
    status = clReleaseMemObject(listBufferA);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseMemObject (listBufferA)\n";
        return SDK_FAILURE; 
    }
	status = clReleaseMemObject(listBufferB);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseMemObject (listBufferB)\n";
        return SDK_FAILURE; 
    }
	status = clReleaseMemObject(mBuffer);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseMemObject (mBuffer)\n";
        return SDK_FAILURE; 
    }
    status = clReleaseCommandQueue(commandQueue);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseCommandQueue\n";
        return SDK_FAILURE;
    }
    status = clReleaseContext(context);
    if(status != CL_SUCCESS)
    {
        std::cout << "Error: In clReleaseContext\n";
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}


/* 
 * Release program resources 
 */
void cleanupHost(void)
{
    if(P != NULL)
    {
        free(P);
        P = NULL;
    }
    if(A != NULL)
    {
        free(A);
        A = NULL;
    }
    if(B != NULL)
    {
        free(B);
        B = NULL;
    }
	if(M != NULL)
    {
        free(M);
        M = NULL;
    }
}

