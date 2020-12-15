
#include <iostream>
#include "color/term.h"
#include "../../MarkovModel/src/model.h"
#include <fstream>


/* Initialize empty markov model
*/
Markov::Model* initialize();

/* initialize model from exported model save file.
*/
Markov::Model* initialize(char* filename);

/* open a dataset file and return ofstream ptr
*/
std::ofstream OpenDatasetFile(char* filename);


/* Read dataset line by line and adjust model with each line
*/
void Train(std::ofstream);

/* Save current model state to file
*/
void Save(char* filename)



int main(int argc, char** argv) {

	terminal t;
	std::cout << TERM_SUCC << terminal::color::RED  <<  "Library loaded." << terminal::color::RESET << std::endl;

	return 0;
}