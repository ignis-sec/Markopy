#include "model.h"
#include "node.h"
#include <fstream>
#include <assert.h>

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(std::ifstream* f) {/*TODO*/ return false; }

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	return this->Import<NodeStorageType>(&importfile);

}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(std::ofstream* f) {/*TODO*/ return false;}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	return this->Export(&exportfile);
}

template <typename NodeStorageType>
NodeStorageType* Markov::Model<NodeStorageType>::RandomWalk() {
	Markov::Node<NodeStorageType>* n = this->starterNode;
	int len = 0;
	NodeStorageType ret[32] = "";
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