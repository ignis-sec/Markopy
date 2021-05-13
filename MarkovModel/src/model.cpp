#pragma once
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

	std::random_device rd;
	generator = new std::default_random_engine(rd()); 
	distribution = std::uniform_int_distribution<long unsigned>(0, 0xffffFFFF);
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
			//std::cout << "Creating new node at start.\n";
		}else {
			srcN = this->nodes.find(src)->second;
		}

		if (this->nodes.find(target) == this->nodes.end()) {
			targetN = new Markov::Node<NodeStorageType>(target);
			this->nodes.insert(std::pair<char, Markov::Node<NodeStorageType>*>(target, targetN));
			//std::cout << "Creating new node at end.\n";
		}
		else {
			targetN = this->nodes.find(target)->second;
		}
		e = srcN->Link(targetN);
		e->AdjustEdge(oc);
		this->edges.push_back(e);
		
		//std::cout << int(srcN->NodeValue()) << " --" << e->EdgeWeight() << "--> " << int(targetN->NodeValue()) << "\n";
		
		
	}

	//std::cout << "Total number of nodes: " << this->nodes.size() << std::endl;
	//std::cout << "Total number of edges: " << this->edges.size() << std::endl;

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
		//std::cout << e->LeftNode()->NodeValue() << "," << e->EdgeWeight() << "," << e->RightNode()->NodeValue() << "\n";
		*f << e->LeftNode()->NodeValue() << "," << e->EdgeWeight() << "," << e->RightNode()->NodeValue() << "\n";
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
NodeStorageType* Markov::Model<NodeStorageType>::RandomWalk(int minSetting, int maxSetting) {  
	std::cout << "Hello\n";
	Markov::Node<NodeStorageType>* n = this->starterNode;
	int len = 0;
	NodeStorageType *ret = new NodeStorageType[64];
	Markov::Node<NodeStorageType> *temp_node;
	while (n != NULL) {
		//n = n->RandomNext();
		temp_node = n->RandomNext();
		//dirty cutoff, needs better solution
		if (len == 60) 
			break;
		if (len > maxSetting) {
			//std::cout<<"MAX ->"<< "node*: " << temp_node << ", len: " << len << "\n";
			break;
		}
			
		if ((temp_node == NULL) && (len < minSetting)) {
			//std::cout << "node*: " << temp_node << ", len: " << len << "\n";
			continue;
		}	

		if (temp_node == NULL)
			break;
		n = temp_node;

		//std::cout << n->NodeValue();
		ret[len++] = n->NodeValue();

		//maximum character length exceeded and stack will overflow.
		//assert(len<32 && "return buffer overflowing, this will segfault if not aborted.");
	}

	//null terminate the string
	ret[len] = 0x00;

	//do something with the generated string
	return ret; //for now
}

template <typename NodeStorageType>
void Markov::Model<NodeStorageType>::AdjustEdge(const NodeStorageType* payload, long int occurrence) {
	NodeStorageType p = payload[0];
	Markov::Node<NodeStorageType>* curnode = this->starterNode;
	Markov::Edge<NodeStorageType> *e;
	int i = 0;
	while (p != 0) {
		e = curnode->FindEdge(p);
		e->AdjustEdge(occurrence);
		curnode = e->RightNode();
		p = payload[++i];
	}

	e = curnode->FindEdge('\xff');
	e->AdjustEdge(occurrence);
	return;
}

