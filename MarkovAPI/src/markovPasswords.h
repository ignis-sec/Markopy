#pragma once
#include "threadSharedListHandler.h"
#include "MarkovModel/src/model.h"


/** @brief Namespace for the MarkovPasswords API
*/
namespace Markov::API{

	/** @brief Markov::Model with char represented nodes. 
	 * 
	 * Includes wrappers for Markov::Model and additional helper functions to handle file I/O
	 * 
	 * This class is an extension of Markov::Model<char>, with higher level abstractions such as train and generate.
	 * 
	*/
	class MarkovPasswords : public Markov::Model<char>{
	public:

		/** @brief Initialize the markov model from MarkovModel::Markov::Model.
		 * 
		 * Parent constructor. Has no extra functionality.
		*/
		MarkovPasswords();

		/** @brief Initialize the markov model from MarkovModel::Markov::Model, with an import file.
		 *
		 * This function calls the Markov::Model::Import on the filename to construct the model.
		 * Same thing as creating and empty model, and calling MarkovPasswords::Import on the filename.
		 * 
		 * @param filename - Filename to import
		 * 
		 * 
		 * @b Example @b Use: Construction via filename
		 * @code{.cpp}
		 * MarkovPasswords mp("test.mdl");
		 * @endcode
		*/
		MarkovPasswords(const char* filename);

		/** @brief Open dataset file and return the ifstream pointer
		 * @param filename - Filename to open
		 * @return ifstream* to the the dataset file
		*/
		std::ifstream* OpenDatasetFile(const char* filename);


		/** @brief Train the model with the dataset file.
		 * @param datasetFileName - Ifstream* to the dataset. If null, use class member
		 * @param delimiter - a character, same as the delimiter in dataset content
		 * @param threads - number of OS threads to spawn
		 * 
		 * @code{.cpp}
		 * Markov::API::MarkovPasswords mp;
		 * mp.Import("models/2gram.mdl");
		 * mp.Train("password.corpus");
		 * @endcode
		*/
		void Train(const char* datasetFileName, char delimiter, int threads);



		/** @brief Export model to file.
		* @param filename - Export filename.
		* @return std::ofstream* of the exported file.
		*/
		std::ofstream* Save(const char* filename);

		/** @brief Call Markov::Model::RandomWalk n times, and collect output.
		 * 
		 * Generate from model and write results to a file. 
		 * a much more performance-optimized method. FastRandomWalk will reduce the runtime by %96.5 on average.
		 * 
		 * @deprecated See Markov::API::MatrixModel::FastRandomWalk for more information.
		 * @param n - Number of passwords to generate.
		 * @param wordlistFileName - Filename to write to
		 * @param minLen - Minimum password length to generate
		 * @param maxLen - Maximum password length to generate
		 * @param threads - number of OS threads to spawn
		*/
		void Generate(unsigned long int n, const char* wordlistFileName, int minLen=6, int maxLen=12, int threads=20);

		/** @brief Buff expression of some characters in the model
		 * @param str A string containing all the characters to be buffed
		 * @param multiplier A constant value to buff the nodes with.
         * @param bDontAdjustSelfEdges Do not adjust weights if target node is same as source node
		 * @param bDontAdjustExtendedLoops Do not adjust if both source and target nodes are in first parameter
		*/
		void Buff(const char* str, double multiplier, bool bDontAdjustSelfLoops=true, bool bDontAdjustExtendedLoops=false);
		

	private:

		/** @brief A single thread invoked by the Train function.
		 * @param listhandler - Listhandler class to read corpus from
		 * @param delimiter - a character, same as the delimiter in dataset content
		 * 
		*/
		void TrainThread(Markov::API::Concurrency::ThreadSharedListHandler *listhandler, char delimiter);

		/** @brief A single thread invoked by the Generate function.
		 * 
		 * @b DEPRECATED: See Markov::API::MatrixModel::FastRandomWalkThread for more information. This has been replaced with 
		 * a much more performance-optimized method. FastRandomWalk will reduce the runtime by %96.5 on average.
		 * 
		 * @param outputLock - shared mutex lock to lock during output operation. Prevents race condition on write.
		 * @param n number of lines to be generated by this thread
		 * @param wordlist wordlistfile
		 * @param minLen - Minimum password length to generate
		 * @param maxLen - Maximum password length to generate
		 * 
		*/
		void GenerateThread(std::mutex *outputLock, unsigned long int n, std::ofstream *wordlist, int minLen, int maxLen);
		std::ifstream* datasetFile; /** @brief	Dataset file input of our system	*/
		std::ofstream* modelSavefile; /** @brief	File to save model  of our system	*/
		std::ofstream* outputFile; /** @brief	Generated output  file  of our system	*/
	};



};
