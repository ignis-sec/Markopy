
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
        __host__ void ListCudaDevices();

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
         * @code{.txt}
         * char *da, a = "test";
         * cudastatus = cudaMalloc((char **)&da, 5*sizeof(char*));
         * CudaCheckNotifyErr(cudastatus, "Failed to allocate VRAM for *da.\n");
		 * @endcode
		*/
        __host__ int CudaCheckNotifyErr(cudaError_t _status, const char* msg);

    private:
    };
};