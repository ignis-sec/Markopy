#include "model.h"
#include <fstream>
#include <assert.h>

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


void Markov::Model::RandomWalk() {
	Markov::Node* n = this->starterNode;
	int len = 0;
	char ret[32] = "";
	while (n != NULL) {
		n = n->RandomNext();
		ret[len++] = n->value();
		assert(len<32 && "return buffer overflowing, this will segfault if not aborted.");
	}
	ret[len] == '\0';

	//do something with the generated string
}