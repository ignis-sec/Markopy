#include "markovPasswords.h"

namespace Markov::API{

    class ModelMatrix : public Markov::API::MarkovPasswords{
    public:
        ModelMatrix();


        void ConstructMatrix();
        void DumpJSON();
        void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen=6, int maxLen=12, int threads=20);

    protected:
        void FastRandomWalkThread(unsigned long int n, const char* wordlistFileName, int minLen=6, int maxLen=12);
        char** edgeMatrix;
        long int **valueMatrix;
        int matrixSize;
        char* matrixIndex;
        long int *totalEdgeWeights;
    };



};