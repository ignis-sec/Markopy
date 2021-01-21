
#include "../../MarkovModel/src/MarkovModel.h"

/** @brief Markov::Model with char represented nodes.
* 
* Includes wrappers for Markov::Model and additional helper functions to handle file I/O
*/
class MarkovPasswords : public Markov::Model<unsigned char>{
public:

	/** @brief Initialize the markov model from MarkovModel::Markov::Model
	*
	*/
	MarkovPasswords();

	/** @brief Initialize the markov model from MarkovModel::Markov::Model, with an import file.
	*
	* This function calls the Markov::Model::Import on the filename to construct the model
	* @param filename - Filename to import
	* @return Pointer to the constructed model.
	*/
	MarkovPasswords(char* filename);

	/** @brief Open dataset file and return the ifstream pointer
	* @param filename - Filename to open
	* @return ifstream* to the the dataset file
	*/
	std::ifstream* OpenDatasetFile(char* filename);


	/** @brief Train the model with the dataset file.
	* @param dataset - Ifstream* to the dataset. If null, use class member
	*/
	void Train(std::ifstream* datasetFile);

	/** @brief Export model to file.
	* @param filename - Export filename.
	* @return std::ofstream* of the exported file.
	*/
	std::ofstream* Save(char* filename);

	/** @brief Call Markov::Model::RandomWalk n times, and collect output.
	* 
	* Write the data to this->outputfile
	* 
	* @param n - Number of passwords to generate.
	* @return std::ofstream* of the output file.
	*/
	std::ofstream* Generate(unsigned long int n);

private:
	std::ifstream* datasetFile;
	std::ofstream* modelSavefile;
	std::ofstream* outputFile;
};