/** @file cudaModelMatrix.cu
 * @brief CUDA accelerated extension of Markov::API::ModelMatrix
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::API::CUDA::CUDAModelMatrix
 */

#include "cudaModelMatrix.h"
#include "cudarandom.h"


#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

    /*__host__ char*  Markov::API::CUDA::CUDAModelMatrix::AllocVRAMOutputBuffer(long int n, long int singleGenMaxLen, long int CUDAKernelGridSize,long int sizePerGrid){
        cudaError_t cudastatus;
        cudastatus = cudaMalloc((char **)&this->device_outputBuffer1, CUDAKernelGridSize*sizePerGrid);
        CudaCheckNotifyErr(cudastatus, "Failed to allocate VRAM buffer. (Possibly out of VRAM.)");
        
        return this->device_outputBuffer1;
    }*/



    __host__ void Markov::API::CUDA::CUDAModelMatrix::FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, bool bFileIO, bool bInfinite){
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


        cudaBlocks = 1024;
        cudaThreads = 256;
        iterationsPerKernelThread = 100;
        alternatingKernels = 2;
        totalOutputPerKernel = (long int)cudaBlocks*(long int)cudaThreads*iterationsPerKernelThread;
        totalOutputPerSync= totalOutputPerKernel*alternatingKernels;
        numberOfPartitions = n/totalOutputPerSync;
        cudaGridSize = cudaBlocks*cudaThreads;
        cudaMemPerGrid = (maxLen+2)*iterationsPerKernelThread;
        cudaPerKernelAllocationSize = cudaGridSize*cudaMemPerGrid;
        this->prepKernelMemoryChannel(alternatingKernels);
        
        unsigned long int leftover = n - (totalOutputPerSync*numberOfPartitions);

        if(bInfinite && !numberOfPartitions) numberOfPartitions=5;
        std::cerr << cudaPerKernelAllocationSize << "\n";

        if(n%totalOutputPerSync) std::cerr << "For optimization, request outputs muliples of "<< totalOutputPerSync << ".\n";

        //start kernelID 1
        this->LaunchAsyncKernel(1, minLen, maxLen);

        for(int i=1;i<numberOfPartitions;i++){
            if(bInfinite) i=0;

            //wait kernelID1 to finish, and start kernelID 0
            cudaStreamSynchronize(this->cudastreams[1]);
            this->LaunchAsyncKernel(0, minLen, maxLen);

            //start memcpy from kernel 1 (block until done)
            this->GatherAsyncKernelOutput(1, bFileIO, wordlist);
            
            //wait kernelID 0 to finish, then start kernelID1
            cudaStreamSynchronize(this->cudastreams[0]);
            this->LaunchAsyncKernel(1, minLen, maxLen);

            //start memcpy from kernel 0 (block until done)
            this->GatherAsyncKernelOutput(0, bFileIO, wordlist);
            
        }

        //wait kernelID1 to finish, and start kernelID 0
        cudaStreamSynchronize(this->cudastreams[1]);
        this->LaunchAsyncKernel(0, minLen, maxLen);
        this->GatherAsyncKernelOutput(1, bFileIO, wordlist);
        cudaStreamSynchronize(this->cudastreams[0]);
        this->GatherAsyncKernelOutput(0, bFileIO, wordlist);

        
        if(!leftover) return;
        alternatingKernels=1;
        std::cerr << "Remaining line count (" << leftover << ") is lower than partition. Adjusting CUDA workload..\n";
        this->iterationsPerKernelThread = leftover/cudaGridSize;
        this->LaunchAsyncKernel(0, minLen, maxLen);
        cudaStreamSynchronize(this->cudastreams[0]);
        this->GatherAsyncKernelOutput(0, bFileIO, wordlist);
        
        leftover -= this->iterationsPerKernelThread*cudaGridSize;
        if(!leftover) return;

        std::cerr << "Remaining line count (" << leftover << ") is lower than minimum possible. Handing over to CPU generation.\n";
        this->iterationsPerKernelThread = leftover/cudaGridSize;

        leftover -= this->iterationsPerKernelThread;

        if(!leftover) return;
        std::cerr << "Remaining " << leftover << " lines are absolutely not worth printing.\n";
        Markov::API::ModelMatrix::ConstructMatrix();
        Markov::API::ModelMatrix::FastRandomWalk(leftover, &wordlist, minLen, maxLen, 1, bFileIO);

    }
    
    __host__ void Markov::API::CUDA::CUDAModelMatrix::prepKernelMemoryChannel(int numberOfStreams){
        
        this->cudastreams = new cudaStream_t[numberOfStreams];
        for(int i=0;i<numberOfStreams;i++)
            cudaStreamCreate(&this->cudastreams[i]);

        this-> outputBuffer = new char*[numberOfStreams];
        for(int i=0;i<numberOfStreams;i++)
            this->outputBuffer[i]= new char[cudaPerKernelAllocationSize];

        cudaError_t cudastatus;
        this-> device_outputBuffer = new char*[numberOfStreams];
            for(int i=0;i<numberOfStreams;i++){
                cudastatus = cudaMalloc((char**)&(device_outputBuffer[i]), cudaPerKernelAllocationSize);
                CudaCheckNotifyErr(cudastatus, "Failed to establish memory channel. Possibly out of VRAM?");
            }

        this-> device_seeds = new unsigned long*[numberOfStreams];
        for(int i=0;i<numberOfStreams;i++){
            Markov::API::CUDA::Random::Marsaglia *MEarr = new Markov::API::CUDA::Random::Marsaglia[cudaGridSize];
            this->device_seeds[i] = Markov::API::CUDA::Random::Marsaglia::MigrateToVRAM(MEarr, cudaGridSize);
            delete[] MEarr;
        }

    }

    __host__ void Markov::API::CUDA::CUDAModelMatrix::LaunchAsyncKernel(int kernelID, int minLen, int maxLen){

        //if(kernelID == 0);// cudaStreamSynchronize(this->cudastreams[2]);
        //else cudaStreamSynchronize(this->cudastreams[kernelID-1]);
        FastRandomWalkCUDAKernel<<<cudaBlocks,cudaThreads,0, this->cudastreams[kernelID]>>>(iterationsPerKernelThread, minLen, maxLen, this->device_outputBuffer[kernelID], this->device_matrixIndex,
            this->device_totalEdgeWeights, this->device_valueMatrix, this->device_edgeMatrix, this->matrixSize, cudaMemPerGrid, this->device_seeds[kernelID]);
        //std::cerr << "Started kernel" << kernelID << "\n";
    }

    __host__ void Markov::API::CUDA::CUDAModelMatrix::GatherAsyncKernelOutput(int kernelID, bool bFileIO, std::ofstream &wordlist){     
        cudaMemcpy(this->outputBuffer[kernelID],this->device_outputBuffer[kernelID],cudaPerKernelAllocationSize, cudaMemcpyDeviceToHost);
        //std::cerr << "Kernel" << kernelID << " output copied\n";
        if(bFileIO){
            for(long int j=0;j<cudaPerKernelAllocationSize;j+=cudaMemPerGrid){
                wordlist << &this->outputBuffer[kernelID][j];
            }
        }else{
            for(long int j=0;j<cudaPerKernelAllocationSize;j+=cudaMemPerGrid){
                std::cout << &this->outputBuffer[kernelID][j];
            }
        }
    }

    __global__ void FastRandomWalkCUDAKernel(unsigned long int n, int minLen, int maxLen, char* outputBuffer,
    char* matrixIndex, long int* totalEdgeWeights, long int* valueMatrix, char *edgeMatrix, int matrixSize, int memoryPerKernelGrid, unsigned long *seed){
        
        int kernelWorkerIndex = threadIdx.x + blockIdx.x * blockDim.x;

        if(n==0) return;

        char* e;
        int index = 0;
        char next;
        int len=0;
        long int selection;
        char cur;
        long int bufferctr = 0;
        unsigned long int *x,*y,*z,t;
        char* res = &outputBuffer[kernelWorkerIndex*memoryPerKernelGrid];
        x=&seed[kernelWorkerIndex*3];
        y=&seed[kernelWorkerIndex*3+1];
        z=&seed[kernelWorkerIndex*3+2];
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
                *x ^= *x << 16;
                *x ^= *x >> 5;
                *x ^= *x << 1;

                t = *x;
                *x = *y;
                *y = *z;
                *z = t ^ *x ^ *y;
                selection = *z % totalEdgeWeights[index];
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