#include "cudaModelMatrix.h"
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaMarkovPasswords/src/cudarandom.h"

using Markov::API::CUDA::CUDADeviceController;

namespace Markov::API::CUDA{
    __host__ void Markov::API::CUDA::CUDAModelMatrix::MigrateMatrix(){
        cudaError_t cudastatus;

        cudastatus = cudaMalloc((char**)&(this->device_matrixIndex), 
            this->matrixSize*sizeof(char));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize device_matrixIndex.");

        cudastatus = cudaMalloc((long int **)&(this->device_totalEdgeWeights), this->matrixSize*sizeof(long int));
        CudaCheckNotifyErr(cudastatus, "Cuda failed to initialize device_totalEdgeWeights.");

        cudastatus = cudaMemcpy(this->device_matrixIndex, this->matrixIndex, 
            this->matrixSize*sizeof(char), cudaMemcpyHostToDevice);
        CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (Index)");

        cudastatus = cudaMemcpy(this->device_totalEdgeWeights, this->totalEdgeWeights, 
            this->matrixSize*sizeof(long int), cudaMemcpyHostToDevice);
        CudaCheckNotifyErr(cudastatus, "Cuda failed to copy to device memory. (Total Edge Values)");

        cudastatus = CudaMigrate2DFlat<char>(
            &(this->device_edgeMatrix), this->edgeMatrix, this->matrixSize, this->matrixSize);
        CudaCheckNotifyErr(cudastatus, "    Cuda failed to initialize edge matrix.");

        cudastatus = CudaMigrate2DFlat<long int>(
            &(this->device_valueMatrix), this->valueMatrix, this->matrixSize, this->matrixSize);
        CudaCheckNotifyErr(cudastatus, "    Cuda failed to initialize value matrix row.");      

    }

    __host__ char*  Markov::API::CUDA::CUDAModelMatrix::AllocVRAMOutputBuffer(long int n, long int singleGenMaxLen, long int CUDAKernelGridSize,long int sizePerGrid){
        cudaError_t cudastatus;
        cudastatus = cudaMalloc((char **)&this->device_outputBuffer, CUDAKernelGridSize*sizePerGrid);
        CudaCheckNotifyErr(cudastatus, "Failed to allocate VRAM buffer. (Possibly out of VRAM.)");
        
        return this->device_outputBuffer;
    }



    __host__ void Markov::API::CUDA::CUDAModelMatrix::FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, bool bFileIO){
        cudaDeviceProp prop;
        int device=0;
        cudaGetDeviceProperties(&prop, device);
        cudaChooseDevice(&device, &prop);
        //std::cout << "Flattening matrix." << std::endl;
        this->FlattenMatrix();
        //std::cout << "Migrating matrix." << std::endl;
        this->MigrateMatrix();
        //std::cout << "Migrated matrix." << std::endl;
        std::ofstream wordlist;	
        if(bFileIO)
            wordlist.open(wordlistFileName);


        int cudaBlocks = 128;
        int cudaThreads = 1024;
        int iterationsPerKernelThread = 500;
        long int totalOutputPerKernel = (long int)cudaBlocks*(long int)cudaThreads*iterationsPerKernelThread;

        //if(n<=totalOutputPerKernel) return FastRandomWalkCUDAKernel<<<1,1>>>(n, minLen, maxLen);
        //else{
        char* buffer;
        int numberOfPartitions = n/totalOutputPerKernel;
        int cudaGridSize = cudaBlocks*cudaThreads;
        
        int cudaMemPerGrid = (maxLen+5)*iterationsPerKernelThread;

        //this->AllocVRAMOutputBuffer(n,maxLen,cudaGridSize, cudaMemPerGrid);
        //std::cout << "Allocated output VRAM." << std::endl;
        
        buffer = new char[cudaGridSize*cudaMemPerGrid];
        cudaMalloc((char**)&this->device_outputBuffer, cudaGridSize*cudaMemPerGrid);
        //std::cout << "Allocated output RAM." << std::endl;
        
        unsigned long *seedChunk;        
        Markov::API::CUDA::Random::Marsaglia *MEarr = new Markov::API::CUDA::Random::Marsaglia[cudaGridSize];
        seedChunk = Markov::API::CUDA::Random::Marsaglia::MigrateToVRAM(MEarr, cudaGridSize);
        //std::cout << "Constucted random devices" << std::endl;

        for(int i=0;i<numberOfPartitions;i++){
            //std::cout << "Running kernel iteration with " << iterationsPerKernelThread << " generations each." << std::endl;
            FastRandomWalkCUDAKernel<<<cudaBlocks,cudaThreads>>>(iterationsPerKernelThread, minLen, maxLen, this->device_outputBuffer, this->device_matrixIndex,
            this->device_totalEdgeWeights, this->device_valueMatrix, this->device_edgeMatrix, this->matrixSize, cudaMemPerGrid, seedChunk);
            
            //std::cout << "Waiting kernel to finish." << std::endl;
            //std::cout << "Iteration done. Retrieving output buffer." << std::endl;
            //cudaDeviceSynchronize();
            cudaMemcpy(buffer,this->device_outputBuffer,cudaGridSize*cudaMemPerGrid, cudaMemcpyDeviceToHost);
            if(bFileIO){
                for(long int j=0;j<cudaGridSize*cudaMemPerGrid;j+=cudaMemPerGrid){
                    wordlist << &buffer[j];
                }
            }else{
                for(long int j=0;j<cudaGridSize*cudaMemPerGrid;j+=cudaMemPerGrid){
                    std::cout << &buffer[j];
                }
            }
            
        }   


    }

    __global__ void FastRandomWalkCUDAKernel(unsigned long int n, int minLen, int maxLen, char* outputBuffer,
    char* matrixIndex, long int* totalEdgeWeights, long int* valueMatrix, char *edgeMatrix, int matrixSize, int memoryPerKernelGrid, unsigned long *seed){
        int kernelWorkerIndex = threadIdx.x + blockIdx.x * blockDim.x;
        //outputBuffer[kernelWorkerIndex]='a'+kernelWorkerIndex;
        /*for(int i=0;i<96;i++){
            for(int j=0;j<96;j++){
                if(edgeMatrix[96*i+j]!='\0')
                    outputBuffer[96*i+j] = edgeMatrix[96*i+j];
                else outputBuffer[96*i+j] = '0';
            }   
            outputBuffer[96*(i+1)] = '\n';
        }*/
        
        //return;
        
        if(n==0) return;

        char* e;
        int index = 0;
        char next;
        int len=0;
        long int selection;
        char cur;
        long int bufferctr = 0;
        long int x,y,z,t;
        char* res = &outputBuffer[kernelWorkerIndex*memoryPerKernelGrid];
        x=seed[kernelWorkerIndex*3];
        y=seed[kernelWorkerIndex*3+1];
        z=seed[kernelWorkerIndex*3+2];
        for (int i = 0; i < n; i++) {
            cur=199;
            len=0;
            while (true) {
                e = strchr(matrixIndex, cur, matrixSize);
                index = e - matrixIndex;
                /*selection = Markov::API::CUDA::Random::devrandom(
                    seed[kernelWorkerIndex*3],
                    seed[kernelWorkerIndex*3+1],
                    seed[kernelWorkerIndex*3+2]) % totalEdgeWeights[index];*/
                x ^= x << 16;
                x ^= x >> 5;
                x ^= x << 1;

                t = x;
                x = y;
                y = z;
                z = t ^ x ^ y;
                selection = z % totalEdgeWeights[index];
            for(int j=0;j<matrixSize-1;j++){
                    selection -= valueMatrix[index*matrixSize + j];
                    if (selection < 0){
                        next = edgeMatrix[index*sizeof(char)*matrixSize + j];
                        break;
                    }
                }

                if (len >= maxLen)  break;
                else if ((next < 0) && (len < minLen)) continue;
                else if (next < 0) break;  
                cur = next;
                res[bufferctr + len++] = cur;
            }
            res[bufferctr + len++] = '\n';
            bufferctr+=len;
        }
        res[bufferctr] = '\0';
    }

    __device__ char* strchr(char* p, char c, int s_len){
       for (;; ++p, s_len--) {
            if (*p ==  c)
                return((char *)p);
            if (!*p)
                return((char *)NULL);
        }
    }

    __host__ void Markov::API::CUDA::CUDAModelMatrix::FlattenMatrix(){
        this->flatEdgeMatrix = new char[this->matrixSize*this->matrixSize];

        this->flatValueMatrix = new long int[this->matrixSize*this->matrixSize];
        for(int i=0;i<this->matrixSize;i++){
            memcpy(&this->flatEdgeMatrix[i*this->matrixSize], this->edgeMatrix[i], this->matrixSize );
            memcpy(&this->flatValueMatrix[i*this->matrixSize], this->valueMatrix[i], this->matrixSize*sizeof(long int) );
        }
    }


};