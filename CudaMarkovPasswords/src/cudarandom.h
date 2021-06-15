#pragma once
#include "MarkovModel/src/random.h"
#include "CudaMarkovPasswords/src/cudaDeviceController.h"

/** @brief Namespace for Random engines operable under __device__ space.
*/
namespace Markov::API::CUDA::Random{

	/** @brief Extension of Markov::Random::Marsaglia which is capable o working on __device__ space.
	*/
    class Marsaglia : public Markov::Random::Marsaglia, public CUDADeviceController{
	public:

		/** @brief Migrate a Marsaglia[] to VRAM as seedChunk
		 * @param MEarr Array of Marsaglia Engines
		 * @param gridSize GridSize of the CUDA Kernel, aka size of array
		 * @returns pointer to the resulting seed chunk in device VRAM.
		*/
		static unsigned long* MigrateToVRAM(Markov::API::CUDA::Random::Marsaglia *MEarr, long int gridSize){
			cudaError_t cudastatus;
			unsigned long* seedChunk;
			cudastatus = cudaMalloc((unsigned long**)&seedChunk, gridSize*3*sizeof(unsigned long));
			CudaCheckNotifyErr(cudastatus, "Failed to allocate seed buffer");
			unsigned long *temp = new unsigned long[gridSize*3];
			for(int i=0;i<gridSize;i++){
				temp[i*3]   = MEarr[i].x;
				temp[i*3+1] = MEarr[i].y;
				temp[i*3+2] = MEarr[i].z;
			}
			//for(int i=0;i<gridSize*3;i++) std::cout << temp[i] << "\n";
			cudaMemcpy(seedChunk, temp, gridSize*3*sizeof(unsigned long), cudaMemcpyHostToDevice);
			CudaCheckNotifyErr(cudastatus, "Failed to memcpy seed buffer.");
			return seedChunk;
		}
	};

	/** @brief Marsaglia Random Generation function operable in __device__ space
	 * @param x marsaglia internal x. Not constant, (ref)
	 * @param y marsaglia internal y. Not constant, (ref)
	 * @param z marsaglia internal z. Not constant, (ref)
	 * @returns returns z
	*/
	__device__ unsigned long devrandom(unsigned long &x, unsigned long &y, unsigned long &z){	
		unsigned long t;
		x ^= x << 16;
		x ^= x >> 5;
		x ^= x << 1;

		t = x;
		x = y;
		y = z;
		z = t ^ x ^ y;

		return z;
	}
};
