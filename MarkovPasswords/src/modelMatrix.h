#include "markovPasswords.h"
#include <mutex>

namespace Markov::API{

    /** @brief Class to flatten and reduce Markov::Model to a Matrix
     * 
     * Matrix level operations can be used for Generation events, with a significant performance optimization at the cost of O(N) memory complexity (O(1) memory space for slow mode)
     * 
     * To limit the maximum memory usage, each generation operation is partitioned into 50M chunks for allocation. Threads are sychronized and files are flushed every 50M operations.
     * 
    */
    class ModelMatrix : public Markov::API::MarkovPasswords{
    public:
        ModelMatrix();

        /** @brief Construct the related Matrix data for the model.
		 * 
		 * This operation can be used after importing/training to allocate and populate the matrix content.
		 * 
         * this will initialize:
         * char** edgeMatrix -> a 2D array of mapping left and right connections of each edge.
         * long int **valueMatrix -> a 2D array representing the edge weights.
         * int matrixSize -> Size of the matrix, aka total number of nodes.
         * char* matrixIndex -> order of nodes in the model
         * long int *totalEdgeWeights -> total edge weights of each Node.
		*/
        void ConstructMatrix();


        /** @brief Debug function to dump the model to a JSON file.
         * 
         * Might not work 100%. Not meant for production use.
		*/
        void DumpJSON();


        /** @brief Random walk on the Matrix-reduced Markov::Model
		 * 
		 * This has an O(N) Memory complexity. To limit the maximum usage, requests with n>50M are partitioned using Markov::API::ModelMatrix::FastRandomWalkPartition.
         * 
         * If n>50M, threads are going to be synced, files are going to be flushed, and buffers will be reallocated every 50M generations.
         * This comes at a minor performance penalty.
         * 
         * While it has the same functionality, this operation reduces Markov::API::MarkovPasswords::Generate runtime by %96.5
         * 
         * This function has deprecated Markov::API::MarkovPasswords::Generate, and will eventually replace it.
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
        void FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen=6, int maxLen=12, int threads=20, bool bFileIO=true);

    protected:

        /** @brief A single partition of FastRandomWalk event
		 * 
		 * Since FastRandomWalk has to allocate its output buffer before operation starts and writes data in chunks, 
         * large n parameters would lead to huge memory allocations.
         * @b Without @b Partitioning:
         * - 50M results 12 characters max -> 550 Mb Memory allocation
         * 
         * - 5B results  12 characters max -> 55 Gb Memory allocation
         * 
         * - 50B results 12 characters max -> 550GB Memory allocation
         * 
         * Instead, FastRandomWalk is partitioned per 50M generations to limit the top memory need.
         * 
         * @param mlock - mutex lock to distribute to child threads
         * @param wordlist - Reference to the wordlist file to write to 
		 * @param n - Number of passwords to generate.
		 * @param wordlistFileName - Filename to write to
		 * @param minLen - Minimum password length to generate
		 * @param maxLen - Maximum password length to generate
		 * @param threads - number of OS threads to spawn
         * @param bFileIO - If false, filename will be ignored and will output to stdout.
         * 
         * 
		*/
        void FastRandomWalkPartition(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, bool bFileIO, int threads);
        
        /** @brief A single thread of a single partition of FastRandomWalk
		 * 
		 * A FastRandomWalkPartition will initiate as many of this function as requested.
         * 
         * This function contains the bulk of the generation algorithm.
         * 
         * @param mlock - mutex lock to distribute to child threads
         * @param wordlist - Reference to the wordlist file to write to 
		 * @param n - Number of passwords to generate.
		 * @param wordlistFileName - Filename to write to
		 * @param minLen - Minimum password length to generate
		 * @param maxLen - Maximum password length to generate
         * @param id - @b DEPRECATED Thread id - No longer used
         * @param bFileIO - If false, filename will be ignored and will output to stdout.
         * 
         * 
		*/
        void FastRandomWalkThread(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, int id, bool bFileIO);
        
		/**
			@brief 2-D Character array for the edge Matrix (The characters of Nodes)
		*/
		char** edgeMatrix;

		/**
			@brief 2-d Integer array for the value Matrix (For the weights of  Edges)
		*/
        long int **valueMatrix;

		/**
			@brief to hold Matrix size
		*/
        int matrixSize;

		/**
			@brief to hold the Matrix index (To hold the orders of 2-D arrays')
		*/
        char* matrixIndex;

		/**
			@brief Array of the Total Edge Weights
		*/
        long int *totalEdgeWeights;
    };



};