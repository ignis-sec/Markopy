
#pragma once
#include <iostream>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/** @brief Namespace for objects requiring CUDA libraries.
*/
namespace Markov::API::CUDA{
    /** @brief Controller class for CUDA device
     * 
     * This implementation only supports Nvidia devices.
    */
    class CUDADeviceController{
    public:
        /** @brief List CUDA devices in the system.
		 *
		 * This function will print details of every CUDA capable device in the system.
         * 
         * @b Example @b output:
         * @code{.txt}
         * Device Number: 0
         * Device name: GeForce RTX 2070
         * Memory Clock Rate (KHz): 7001000
         * Memory Bus Width (bits): 256
         * Peak Memory Bandwidth (GB/s): 448.064
         * Max Linear Threads: 1024
		 * @endcode
		*/
        __host__ static void ListCudaDevices();

    protected:
        /** @brief Check results of the last operation on GPU.
         * 2M for *da.");
		 * @endcode
		*/
        __host__ static int CudaCheckNotifyErr(cudaError_t _status, const char* msg, bool bExit=true);


        template <typename T>
        __host__ static cudaError_t CudaMalloc2DToFlat(T** dst, int row, int col){
            cudaError_t cudastatus = cudaMalloc((T **)dst, row*col*sizeof(T));
            CudaCheckNotifyErr(cudastatus, "cudaMalloc Failed.", false);
            return cudastatus;
        }

        template <typename T>
        __host__ static cudaError_t CudaMemcpy2DToFlat(T* dst, T** src, int row, int col){
            T* tempbuf = new T[row*col];
            for(int i=0;i<row;i++){
                memcpy(&(tempbuf[row*i]), src[i], col);
            }
            return cudaMemcpy(dst, tempbuf, row*col*sizeof(T), cudaMemcpyHostToDevice);
            
        }

        template <typename T>
        __host__ static cudaError_t CudaMigrate2DFlat(T** dst, T** src, int row, int col){
            cudaError_t cudastatus;
            cudastatus = CudaMalloc2DToFlat<T>(dst, row, col);
            if(cudastatus!=cudaSuccess){
                CudaCheckNotifyErr(cudastatus, "  CudaMalloc2DToFlat Failed.", false);
                return cudastatus;
            }
            cudastatus = CudaMemcpy2DToFlat<T>(*dst,src,row,col);
            CudaCheckNotifyErr(cudastatus, "  CudaMemcpy2DToFlat Failed.", false);
            return cudastatus;
        }
        

    private:
    };
};