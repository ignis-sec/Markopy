#include "MarkovPasswords/src/modelMatrix.h"


namespace Markov::API::CUDA{
    class CUDAModelMatrix : public ModelMatrix{
    public:
        __host__ void MigrateMatrix();
        __host__ void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads, bool bFileIO);
        __host__ void ListCudaDevices();

    protected:
        __host__ int CudaCheckNotifyErr(cudaError_t _status, const char* msg);
        __host__ void RetrieveCudaBuffer(/*TODO*/);


    private:
        char** device_edgeMatrix;
        long int **device_valueMatrix;
        char *device_matrixIndex;
        long int *device_totalEdgeWeights;
    };
};