#include "cudaDeviceController.h"
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace Markov::API::CUDA{
    __host__ void Markov::API::CUDA::CUDADeviceController::ListCudaDevices() { //list cuda Capable devices on host.
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device Number: " <<  i  << "\n";
            std::cout << "Device name: " << prop.name << "\n";
            std::cout << "Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
            std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
            std::cout << "Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << "\n";
            std::cout << "Max Linear Threads: " << prop.maxThreadsDim[0] << "\n";
            
        }
    }

    __host__ int Markov::API::CUDA::CUDADeviceController::CudaCheckNotifyErr(cudaError_t _status, const char* msg) {
        if (_status != cudaSuccess) {
            std::cerr << "\033[1;31m" << _status << ": " << cudaGetErrorString(_status) << "-> "  << msg << "\033[0m" << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return 0;
    }
    __global__ static void FastRandomWalkPartition(unsigned long int n, int minLen, int maxLen, bool bFileIO, int threads){

    }
};