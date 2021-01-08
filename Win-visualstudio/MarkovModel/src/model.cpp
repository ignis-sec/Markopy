#include "model.h"
#include <fstream>
#include <assert.h>

/* Import model structure from a savefile
*/
bool Markov::Model::Import(std::ifstream* f) {/*TODO*/ }

//import wrapper for filename
bool Markov::Model::Import(char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	this->Import(&importfile);
}

/* Export model structure to a savefile
*/
bool Markov::Model::Export(std::ofstream* f) {/*TODO*/}

//export wrapper for filename
bool Markov::Model::Export(char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	this->Export(&exportfile);
}


//Randomwalk implementation.
//Continue until RandomNext returns null
//Append each walked nodes value to the return buffer
//TODO: Output/process the walked string
void Markov::Model::RandomWalk() {
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
}