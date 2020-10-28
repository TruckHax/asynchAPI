////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system
#include <iostream>
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

// Questions:
//
// how many streaming multiprocessors (SM's) are on my GeForce GTX 1050 Ti? the GTX 1080Ti has 28 SM's
//
// https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/
//

// CUDA class - https://stackoverflow.com/questions/6978643/cuda-and-classes
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__    // __CUDACC__ is defined by NVCC when it is compiling CUDA files
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Foo {
public:
    CUDA_CALLABLE_MEMBER Foo() {}
    CUDA_CALLABLE_MEMBER ~Foo() {}
    CUDA_CALLABLE_MEMBER void aMethod() {}
};

/* very simple sample
__global__ void mykernel(void) {
}

int main(void) {
    mykernel << <1, 1 >> > ();
    printf("Hello World!\n");
    return 0;
} */

#ifdef __CUDACC__
__host__ __device__
#endif
int countLeadingZeros(unsigned int a)
{
#if defined(__CUDA_ARCH__)
    return __popc(a);
#else
    // Source: http://graphics.stanford.edu/~seander/bithacks.html
    a = a - ((a >> 1) & 0x55555555);
    a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
    return ((a + (a >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
}

// acronyms:
// PTX(Parallel Thread Execution)
// ISA(Instruction Set Architecture)
// SIMT (single-instruction, multiple-thread)


// Introduction to CUDA C/C++
// https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf?ncid=afm-chs-44270&ranMID=44270&ranEAID=msYS1Nvjv4c&ranSiteID=msYS1Nvjv4c-5giwA16DHXNXTVTE9qtW8g

// CUDA Kernels with C++
// https://corecppil.github.io/Meetups/2018-08-18_Core-C++TLV/CUDACpp.pdf

__global__ void addKernel(int* c, int* a, int* b, int n)
{
    // *c = *a + *b;

//    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];  // parallel blocks

//    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];   // parallel threads

    // With M threads per block, a unique index for each thread is given by
//    int index = threadIdx.x + blockIdx.x * blockDim.x;   // where blockDim.x is M
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();
}

#if _MSC_VER >= 1920    // VS 2019 - ?? CUDA_VERSION
__global__ void increment_kernel()
{
    printf("blockIdx x %02d y %02d z %02d threadIdx x %02d y %02d z %02d blockDim x %02d y %02d z %02d gridDim x %02d y %02d z %02d\n",
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}
#else
__global__ void increment_kernel()
{

}
#endif

// https://stackoverflow.com/questions/28044011/cudaoccupancymaxactiveblockspermultiprocessor-is-undefined
__global__ void MyKernel(int* d, int* a, int* b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}


// TODO 
// draw grid with blocks, warps, threads
//

bool correct_output(int* data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }
    }

    return true;
}

#define __MAIN

#ifdef __MAIN
int main(int argc, char* argv[])
{
    int devID = 0;
    int deviceCount = 0;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char**)argv);

    // cudaChooseDevice

    checkCudaErrors(cudaGetDevice(&devID));

    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));

    // cudaGetDeviceFlags
    // cudaDeviceGetAttribute
    // cudaDeviceGetLimit() call stack

    printf("name %s warpSize %d clockRate %d maxThreadsPerBlock %d deviceCount %d\n\n",
        deviceProps.name, deviceProps.warpSize, deviceProps.clockRate, deviceProps.maxThreadsPerBlock, deviceCount);

    int n = 2; // *1024 * 1024;
    int nbytes = n * sizeof(int);

    int value = 26;

    // allocate host memory
    int* a = 0;
    checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int* d_a = 0;
    checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(2, 1, 1);
    dim3 blocks = dim3(4, 2, 2);

    printf("blocks x %02d y %02d z %02d, threads x %02d y %02d z %02d\n\n",
        blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);

    // 32 threads per block works well
    increment_kernel << <blocks, threads, 0, 0 >> > ();  // parallel blocks, threads

    // addKernel<int> << <blocks, 32 >> > (dev_c, dev_a, dev_b);

    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    printf("cudaGetLastError %s\n", cudaGetErrorString(cudaGetLastError()));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
#else
// occupancy is the ratio of the number of active warps per multiprocessor to the maximum number of warps that can be active on the multiprocessor at once
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    return 0;
}
#endif

/* TODO
void launchMyKernel(int* array, int arrayCount)
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        MyKernel, 0, 0);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel << < gridSize, blockSize >> > (array, arrayCount);

    cudaDeviceSynchronize();

    // calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
        MyKernel, blockSize,
        0);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
        (float)(props.maxThreadsPerMultiProcessor /
            props.warpSize);

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
        blockSize, occupancy);
}
*/
