#include "model.h"
#include <fstream>

/* Import model structure from a savefile
*/
bool Markov::Model::Import(std::ifstream* f) {/*TODO*/ }

bool Markov::Model::Import(char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	this->Import(&importfile);
}

/* Export model structure to a savefile
*/

bool Markov::Model::Export(std::ofstream* f) {/*TODO*/}

bool Markov::Model::Export(char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	this->Export(&exportfile);
}