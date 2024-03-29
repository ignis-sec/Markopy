/** @file cudaDeviceController.cu
 * @brief Simple static class for basic CUDA device controls.
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::API::CUDA::CUDADeviceController
 */

#include "cudaDeviceController.h"
#include <iostream>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Markov::API::CUDA{
    __host__ void Markov::API::CUDA::CUDADeviceController::ListCudaDevices() { //list cuda Capable devices on host.
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cerr << "Device Number: " <<  i  << "\n";
            std::cerr << "Device name: " << prop.name << "\n";
            std::cerr << "Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
            std::cerr << "Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
            std::cerr << "Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << "\n";
            std::cerr << "Max Linear Threads: " << prop.maxThreadsDim[0] << "\n";
            
        }
    }

    __host__ int Markov::API::CUDA::CUDADeviceController::CudaCheckNotifyErr(cudaError_t _status, const char* msg, bool bExit) {
        if (_status != cudaSuccess) {
            std::cerr << "\033[1;31m" << msg  << " -> " << cudaGetErrorString(_status)  << " ("<< _status << ")" << "\033[0m" << "\n";
            
            if(bExit) {
                cudaDeviceReset();
                exit(1);
            }
        }
        return 0;
    }

/*
    template <typename T>
    __host__ cudaError_t Markov::API::CUDA::CUDADeviceController::CudaMalloc2DToFlat(T* dst, int row, int col){
        return  cudaMalloc((T **)&dst, row*col*sizeof(T));
    }

    template <typename T>
    __host__ cudaError_t Markov::API::CUDA::CUDADeviceController::CudaMemcpy2DToFlat(T* dst, T** src, int row, int col){
         cudaError_t cudastatus;
         for(int i=0;i<row;i++){
            cudastatus = cudaMemcpy(dst + (i*col*sizeof(T)), 
                src[i], col*sizeof(T), cudaMemcpyHostToDevice);
            if(cudastatus != cudaSuccess) return cudastatus;
        }
        return cudaSuccess;
    }
*/

};