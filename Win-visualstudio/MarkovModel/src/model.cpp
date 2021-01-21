#include "model.h"
#include "node.h"
#include <fstream>
#include <assert.h>
#include <string>
#include <iostream>

template <typename NodeStorageType>
Markov::Model<NodeStorageType>::Model() {
	this->starterNode = new Markov::Node<NodeStorageType>(0);
	this->nodes.insert({ 0, this->starterNode });
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(std::ifstream *f) {
	std::string cell;

	char src;
	char target;
	int oc;
	
	while (std::getline(*f, cell) ) {
		//std::cout << "cell: " << cell << std::endl;
		src = cell[0];
		target = cell[cell.length()-1];
		oc = std::atoi(cell.substr(2, cell.length() - 2).c_str());
		
		
		Markov::Node<NodeStorageType>* srcN;
		Markov::Node<NodeStorageType>* targetN;
		Markov::Edge<NodeStorageType>* e;
		if (this->nodes.find(src) == this->nodes.end()) {
			srcN = new Markov::Node<NodeStorageType>(src);
			this->nodes.insert(std::pair<char, Markov::Node<NodeStorageType>*>(src, srcN));
			std::cout << "Creating new node at start.\n";
		}else {
			srcN = this->nodes.find(src)->second;
		}

		if (this->nodes.find(target) == this->nodes.end()) {
			targetN = new Markov::Node<NodeStorageType>(target);
			this->nodes.insert(std::pair<char, Markov::Node<NodeStorageType>*>(target, targetN));
			std::cout << "Creating new node at end.\n";
		}
		else {
			targetN = this->nodes.find(target)->second;
		}
		e = new Markov::Edge<NodeStorageType>(srcN, targetN);
		this->edges.push_back(e);

		//std::cout << srcN->value() << " --" << e->weight() << "--> " << targetN->value() << "\n";
	}

	std::cout << "Total number of nodes: " << this->nodes.size() << std::endl;
	std::cout << "Total number of edges: " << this->edges.size() << std::endl;

	return true;
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(const char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	return this->Import(&importfile);

}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(std::ofstream* f) {
	Markov::Edge<NodeStorageType>* e;
	for (std::vector<int>::size_type i = 0; i != this->edges.size(); i++) {
		e = this->edges[i];
		*f << e->left()->value() << "," << e->weight() << "," << e->right()->value() << "\n";
	}

	return true;
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(const char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	return this->Export(&exportfile);
}

template <typename NodeStorageType>
NodeStorageType* Markov::Model<NodeStorageType>::RandomWalk() {
	Markov::Node<NodeStorageType>* n = this->starterNode;
	int len = 0;
	NodeStorageType ret[32];
	while (n != NULL) {
		n = n->RandomNext();
		ret[len++] = n->value();

		//maximum character length exceeded and stack will overflow.
		assert(len<32 && "return buffer overflowing, this will segfault if not aborted.");
	}

	//null terminate the string
	ret[len] == NULL;

	//do something with the generated string
	return ret; //for now
}

template <typename NodeStorageType>
void Markov::Model<NodeStorageType>::adjust(NodeStorageType* payload, long int occurrence) {
	NodeStorageType p = payload[0];
	Markov::Node<NodeStorageType>* curnode = this->starterNode;
	Markov::Edge<NodeStorageType> *e;
	while (p != 0) {
		e = curnode->findEdge(p);
		e->adjust(occurrence);
		curnode = e->traverse();
	}

	e = curnode->findEdge(0xff);
	return;
}
