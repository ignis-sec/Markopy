
#include <iostream>
#include "color/term.h"
#include "../../MarkovModel/src/model.h"
#include <fstream>



/** @brief Namespace for password operations derived from MarkovModel.lib
*/
namespace MarkovPasswords{

	/** @brief Initialize the markov model from MarkovModel::Markov::Model
	* @return Pointer to the constructed model.
	*/
	Markov::Model* initialize();

	/** @brief Initialize the markov model from MarkovModel::Markov::Model, with an import file.
	* 
	* This function calls the Markov::Model::Import on the filename to construct the model 
	* @param filename - Filename to import
	* @return Pointer to the constructed model.
	*/
	Markov::Model* initialize(char* filename);

	/** @brief Open dataset file and return the ifstream pointer
	* @param filename - Filename to open
	* @return ifstream* to the the dataset file 
	*/
	std::ifstream* OpenDatasetFile(char* filename);


	/** @brief Train the model with the dataset file.
	* @param dataset - Ifstream* to the dataset
	*/
	void Train(std::ifstream*);

	/** @brief Export model to file.
	* @param filename - Export filename.
	* @return std::ofstream* of the exported file.
	*/
	std::ofstream* Save(char* filename);
}




int main(int argc, char** argv) {

	terminal t;
	std::cout << TERM_SUCC << terminal::color::RED  <<  "Library loaded." << terminal::color::RESET << std::endl;

	return 0;
}