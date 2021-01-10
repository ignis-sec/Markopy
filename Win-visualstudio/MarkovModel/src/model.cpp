#include "model.h"
#include <fstream>
#include <assert.h>

bool Markov::Model::Import(std::ifstream* f) {/*TODO*/ return false; }

bool Markov::Model::Import(char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	return this->Import(&importfile);

}

bool Markov::Model::Export(std::ofstream* f) {/*TODO*/ return false;}

bool Markov::Model::Export(char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	return this->Export(&exportfile);
}

char* Markov::Model::RandomWalk() {
	Markov::Node* n = this->starterNode;
	int len = 0;
	char ret[32] = "";
	while (n != NULL) {
		n = n->RandomNext();
		ret[len++] = n->value();

		//maximum character length exceeded and stack will overflow.
		assert(len<32 && "return buffer overflowing, this will segfault if not aborted.");
	}

	//null terminate the string
	ret[len] == '\0';

	//do something with the generated string
	return ret; //for now
}