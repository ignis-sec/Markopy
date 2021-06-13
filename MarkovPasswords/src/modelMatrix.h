#include "markovPasswords.h"
#include <mutex>

namespace Markov::API{

    class ModelMatrix : public Markov::API::MarkovPasswords{
    public:
        ModelMatrix();


        void ConstructMatrix();
        void DumpJSON();
        void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen=6, int maxLen=12, int threads=20, bool bFileIO=true);

    protected:
        void FastRandomWalkPartition(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, bool bFileIO, int threads);
        void FastRandomWalkThread(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, int id, bool bFileIO);
        char** edgeMatrix;
        long int **valueMatrix;
        int matrixSize;
        char* matrixIndex;
        long int *totalEdgeWeights;
    };



};