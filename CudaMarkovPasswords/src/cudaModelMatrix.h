#include "MarkovPasswords/src/modelMatrix.h"
#include "cudaDeviceController.h"   

/** @brief Namespace for objects requiring CUDA libraries.
*/
namespace Markov::API::CUDA{
    /** @brief Extension of Markov::API::ModelMatrix which is modified to run on GPU devices.
     * 
     * This implementation only supports Nvidia devices.
    */
    class CUDAModelMatrix : public ModelMatrix, public CUDADeviceController{
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
        __host__ void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads, bool bFileIO);

    protected:
        /** @brief Retrieve the result buffer from CUDA kernel.
         * 
         * Done on each partition.
         * 
         * 
		*/
        __host__ void RetrieveCudaBuffer(/*TODO*/);


    private:
        char** device_edgeMatrix;
        long int **device_valueMatrix;
        char *device_matrixIndex;
        long int *device_totalEdgeWeights;
    };
};