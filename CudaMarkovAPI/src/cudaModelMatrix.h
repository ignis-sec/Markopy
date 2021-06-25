/** @file cudaModelMatrix.h
 * @brief CUDA accelerated extension of Markov::API::ModelMatrix
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::API::CUDA::CUDAModelMatrix
 */

#include "MarkovAPI/src/modelMatrix.h"
#include "cudaDeviceController.h"   

/** @brief Namespace for objects requiring CUDA libraries.
*/
namespace Markov::API::CUDA{
    /** @brief Extension of Markov::API::ModelMatrix which is modified to run on GPU devices.
     * 
     * This implementation only supports Nvidia devices.
     * @copydoc Markov::API::ModelMatrix
    */
    class CUDAModelMatrix : public Markov::API::ModelMatrix, public CUDADeviceController{
    public:

        /** @brief Migrate the class members to the VRAM
		 *
		 * Cannot be used without calling Markov::API::ModelMatrix::ConstructMatrix at least once.
         * This function will manage the memory allocation and data transfer from CPU RAM to GPU VRAM.
         * 
         * Newly allocated VRAM pointers are set in the class member variables.
		 * 
		*/
        __host__ void MigrateMatrix();

        /** @brief Flatten migrated matrix from 2d to 1d
		 *
		 * 
		*/
        __host__ void FlattenMatrix();

        /** @brief Random walk on the Matrix-reduced Markov::Model
         * 
         * TODO
		 * 
		 * 
		 * @param n - Number of passwords to generate.
		 * @param wordlistFileName - Filename to write to
		 * @param minLen - Minimum password length to generate
		 * @param maxLen - Maximum password length to generate
		 * @param threads - number of OS threads to spawn
         * @param bFileIO - If false, filename will be ignored and will output to stdout.
         * 
         * 
         * @code{.cpp}
		 * Markov::API::ModelMatrix mp;
		 * mp.Import("models/finished.mdl");
         * mp.FastRandomWalk(50000000,"./wordlist.txt",6,12,25, true);
		 * @endcode
         * 
		*/
        __host__ void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, bool bFileIO, bool bInfinite);

    protected:

        /** @brief Allocate the output buffer for kernel operation
         * 
         * TODO
		 * 
		 * 
		 * @param n - Number of passwords to generate.
		 * @param singleGenMaxLen - maximum string length for a single generation
		 * @param CUDAKernelGridSize - Total number of grid members in CUDA kernel
		 * @param sizePerGrid - Size to allocate per grid member
         * @return pointer to the allocation on VRAM
         * 
         * 
		*/
        __host__ char* AllocVRAMOutputBuffer(long int n, long int singleGenMaxLen, long int CUDAKernelGridSize,long int sizePerGrid);

        __host__ void LaunchAsyncKernel(int kernelID, int minLen, int maxLen);

        __host__ void prepKernelMemoryChannel(int numberOfStreams);

        __host__ void GatherAsyncKernelOutput(int kernelID, bool bFileIO, std::ofstream &wordlist);

    private:

        /** 
			@brief VRAM Address pointer of edge matrix (from modelMatrix.h)
		*/ 
        char* device_edgeMatrix;
		
        /** 
			@brief VRAM Address pointer of value matrix (from modelMatrix.h)
		*/ 
        long int *device_valueMatrix;
		
        /** 
			@brief VRAM Address pointer of matrixIndex (from modelMatrix.h)
		*/ 
        char *device_matrixIndex;
		
        /** 
			@brief VRAM Address pointer of total edge weights (from modelMatrix.h)
		*/ 
        long int *device_totalEdgeWeights;
		
        /** 
			@brief RandomWalk results in device
		*/ 		
        char** device_outputBuffer;
		
        /** 
			@brief RandomWalk  results in host
		*/ 	
        char** outputBuffer;

        /** 
			@brief Adding Edge matrix end-to-end and resize to 1-D array for better perfomance on traversing
		*/ 	
        char* flatEdgeMatrix;
		
        /** 
			@brief Adding Value matrix end-to-end and resize to 1-D array for better perfomance on traversing
		*/ 	
        long int* flatValueMatrix; 

        int cudaBlocks;
        int cudaThreads;
        int iterationsPerKernelThread;
        long int totalOutputPerSync;
        long int totalOutputPerKernel;
        int numberOfPartitions;
        int cudaGridSize;
        int cudaMemPerGrid;
        long int cudaPerKernelAllocationSize;

        int alternatingKernels;

        unsigned long**  device_seeds;

        cudaStream_t *cudastreams;

    };

    /** @brief CUDA kernel for the FastRandomWalk operation
     * 
     * Will be initiated by CPU and continued by GPU (__global__ tag)
     * 
     * 
     * @param n - Number of passwords to generate.
     * @param minlen - minimum string length for a single generation
     * @param maxLen - maximum string length for a single generation
     * @param outputBuffer - VRAM ptr to the output buffer
     * @param matrixIndex - VRAM ptr to the matrix indices
     * @param totalEdgeWeights - VRAM ptr to the totalEdgeWeights array
     * @param valueMatrix - VRAM ptr to the edge weights array
     * @param edgeMatrix - VRAM ptr to the edge representations array
     * @param matrixSize - Size of the matrix dimensions
     * @param memoryPerKernelGrid - Maximum memory usage per kernel grid
     * @param seed - seed chunk to generate the random from (generated & used by Marsaglia)
     * 
     * 
     * 
    */
    __global__ void FastRandomWalkCUDAKernel(unsigned long int n, int minLen, int maxLen, char* outputBuffer,
        char* matrixIndex, long int* totalEdgeWeights, long int* valueMatrix, char *edgeMatrix, 
        int matrixSize, int memoryPerKernelGrid, unsigned long *seed);//, unsigned long mex, unsigned long mey, unsigned long mez);
    

    /** @brief srtchr implementation on __device__ space
     * 
     * Fint the first matching index of a string
     * 
     * 
     * @param p - string to check
     * @param c - character to match
     * @param s_len - maximum string length
     * @returns pointer to the match
    */
    __device__ char* strchr(char* p, char c, int s_len);
    
};

