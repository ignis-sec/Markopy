
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
         * 
         * Check the status returned from cudaMalloc/cudaMemcpy to find failures.
         * 
         * If a failure occurs, its assumed beyond redemption, and exited.
         * @param _status Cuda error status to check 
         * @param msg Message to print in case of a failure
         * @return 0 if successful, 1 if failure.
         * @b Example @b output:
         * @code{.cpp}
         * char *da, a = "test";
         * cudastatus = cudaMalloc((char **)&da, 5*sizeof(char*));
         * CudaCheckNotifyErr(cudastatus, "Failed to allocate VRAM for *da.\n");
		 * @endcode
		*/
        __host__ static int CudaCheckNotifyErr(cudaError_t _status, const char* msg, bool bExit=true);

        
        /** @brief Malloc a 2D array in device space
         * 
         * This function will allocate enough space on VRAM for flattened 2D array.
         * 
         * @param dst destination pointer
         * @param row row size of the 2d array
         * @param col column size of the 2d array
         * @return cudaError_t status of the cudaMalloc operation
         * 
         * @b Example @b output:
         * @code{.cpp}
         *   cudaError_t cudastatus;
         *   char* dst;
         *   cudastatus = CudaMalloc2DToFlat<char>(&dst, 5, 15);
         *   if(cudastatus!=cudaSuccess){
         *       CudaCheckNotifyErr(cudastatus, "  CudaMalloc2DToFlat Failed.", false);
         *   }
		 * @endcode
		*/
        template <typename T>
        __host__ static cudaError_t CudaMalloc2DToFlat(T** dst, int row, int col){
            cudaError_t cudastatus = cudaMalloc((T **)dst, row*col*sizeof(T));
            CudaCheckNotifyErr(cudastatus, "cudaMalloc Failed.", false);
            return cudastatus;
        }


        /** @brief Memcpy a 2D array in device space after flattening
         * 
         * Resulting buffer will not be true 2D array.
         * 
         * @param dst destination pointer
         * @param rc  source pointer
         * @param row row size of the 2d array
         * @param col column size of the 2d array
         * @return cudaError_t status of the cudaMalloc operation
         * 
         * @b Example @b output:
         * @code{.cpp}
         *   cudaError_t cudastatus;
         *   char* dst;
         *   cudastatus = CudaMalloc2DToFlat<char>(&dst, 5, 15);
         *   CudaCheckNotifyErr(cudastatus, "  CudaMalloc2DToFlat Failed.", false);
         *   cudastatus = CudaMemcpy2DToFlat<char>(*dst,src,15,15);
         *   CudaCheckNotifyErr(cudastatus, "  CudaMemcpy2DToFlat Failed.", false);
		 * @endcode
		*/
        template <typename T>
        __host__ static cudaError_t CudaMemcpy2DToFlat(T* dst, T** src, int row, int col){
            T* tempbuf = new T[row*col];
            for(int i=0;i<row;i++){
                memcpy(&(tempbuf[row*i]), src[i], col);
            }
            return cudaMemcpy(dst, tempbuf, row*col*sizeof(T), cudaMemcpyHostToDevice);
            
        }

        /** @brief Both malloc and memcpy a 2D array into device VRAM.
         * 
         * Resulting buffer will not be true 2D array.
         * 
         * @param dst destination pointer
         * @param rc  source pointer
         * @param row row size of the 2d array
         * @param col column size of the 2d array
         * @return cudaError_t status of the cudaMalloc operation
         * 
         * @b Example @b output:
         * @code{.cpp}
         *   cudaError_t cudastatus;
         *   char* dst;
         *   cudastatus = CudaMigrate2DFlat<long int>(
         *      &dst, this->valueMatrix, this->matrixSize, this->matrixSize);
         *   CudaCheckNotifyErr(cudastatus, "    Cuda failed to initialize value matrix row.");
		 * @endcode
		*/
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