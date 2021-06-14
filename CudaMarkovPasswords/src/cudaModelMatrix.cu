#include "cudaModelMatrix.h"
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Markov::API::CUDA{
    __host__ void Markov::API::CUDA::CUDAModelMatrix::MigrateMatrix(){
        cudaError_t cudastatus;
        cudastatus = cudaMalloc((char ***)&this->device_edgeMatrix, this->matrixSize*sizeof(char*));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize edge matrix.\n");
        for(int i=0;i<this->matrixSize;i++){
            cudastatus = cudaMalloc((char **)&this->device_edgeMatrix[i],this->matrixSize*sizeof(char));
            CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize edge matrix row.\n");
        }

        cudastatus = cudaMalloc((long int ***)&this->device_valueMatrix, this->matrixSize*sizeof(long int*));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize value matrix row.\n");
        for(int i=0;i<this->matrixSize;i++){
            cudastatus = cudaMalloc((char **)&this->device_valueMatrix[i],this->matrixSize*sizeof(long int));
            CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize value matrix row.\n");
        }

        cudastatus = cudaMalloc((char**)&this->device_matrixIndex, this->matrixSize*sizeof(char));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize device_matrixIndex.\n");

        cudastatus = cudaMalloc((long int **)&this->device_totalEdgeWeights, this->matrixSize*sizeof(long int));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize device_totalEdgeWeights.\n");

        for(int i=0;i<this->matrixSize;i++){
            cudastatus = cudaMemcpy(this->device_edgeMatrix[i], this->edgeMatrix[i], this->matrixSize*sizeof(char), cudaMemcpyHostToDevice);
            CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (edge matrix)\n");
        }

        for(int i=0;i<this->matrixSize;i++){
            cudastatus = cudaMemcpy(this->device_valueMatrix[i], this->valueMatrix[i], this->matrixSize*sizeof(long int), cudaMemcpyHostToDevice);
            CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (value matrix)\n");
        }

        cudastatus = cudaMemcpy(this->device_matrixIndex, this->matrixIndex, this->matrixSize*sizeof(char), cudaMemcpyHostToDevice);
        CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (Index)\n");

        cudastatus = cudaMemcpy(this->device_totalEdgeWeights, this->totalEdgeWeights, this->matrixSize*sizeof(long int), cudaMemcpyHostToDevice);
        CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (Total Edge Values)\n");
        

    }
    __host__ void Markov::API::CUDA::CUDAModelMatrix::RetrieveCudaBuffer(){}
    __host__ void Markov::API::CUDA::CUDAModelMatrix::FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads, bool bFileIO){

    }
    __host__ void Markov::API::CUDA::CUDAModelMatrix::ListCudaDevices() { //list cuda Capable devices on host.
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

    __host__ int Markov::API::CUDA::CUDAModelMatrix::CudaCheckNotifyErr(cudaError_t _status, const char* msg) {
        if (_status != cudaSuccess) {
            std::cerr << "\033[1;31m" << _status << ": " << cudaGetErrorString(_status) << "-> "  << msg << "\033[0m" << "\n";
            cudaDeviceReset();
            exit(1);
        }
        return 0;
    }
    __global__ static void FastRandomWalkPartition(unsigned long int n, int minLen, int maxLen, bool bFileIO, int threads){

    }
}