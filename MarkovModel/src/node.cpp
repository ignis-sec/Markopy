#pragma once
#include "node.h"
#include <assert.h>
#include <iostream>
#include <stdexcept> // To use runtime_error

template <typename storageType>
Markov::Node<storageType>::Node(storageType _value) {
	this->_value = _value;
};

template <typename storageType>
Markov::Node<storageType>::Node() {
	this->_value = 0;
	this->total_edge_weights = 0;
};

template <typename storageType>
unsigned char Markov::Node<storageType>::value() {
	return _value;
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::Link(Markov::Node<storageType>* n) {
	Markov::Edge<storageType> *v = new Markov::Edge<storageType>(this, n);
	this->UpdateEdges(v);
	return v;
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::Link(Markov::Edge<storageType>* v) {
	v->set_left(this);
	this->UpdateEdges(v);
	return v;
}

template <typename storageType>
Markov::Node<storageType>* Markov::Node<storageType>::RandomNext() {

	//get a random value in range of total_vertice_weight
	int rnd = distribution(generator);// distribution(generator);

	int selection = rnd % this->total_edge_weights; //add division by zero execption handling //replace with next lines while not empty file
	/*if(this->total_edge_weights==0)
		throw std::runtime_error("Math error: Attempted to divide by zero\n");
	try {
		int selection = rnd % this->total_edge_weights;
	}
	catch (std::runtime_error e) {

		// prints that exception has occurred
		// calls the what function using object of
		// runtime_error class
		std::cout << "Exception occurred" << std::endl
			<< e.what();
	}*/


	//make absolute, no negative modulus values wanted
	selection = (selection>=0)? selection : (selection + this->total_edge_weights);

	//iterate over the Edge map
	//Subtract the Edge weight from the selection at each Edge
	//when selection goes below 0, pick that node 
	//(Fast random selection with weight bias)
	//std::cout << "Rand: " << rnd << "\n";
	//std::cout << "Total: " << this->total_edge_weights << "\n";
	//std::cout << "Total edges: " << this->edges.size() << "\n";
	for ( std::pair<unsigned char,Markov::Edge<storageType>*> const& x : this->edges) {
		//std::cout << selection << "\n";
		selection -= x.second->weight();
		//std::cout << selection << "\n";
		if (selection < 0) return x.second->traverse();
	}

	//if this assertion is reached, it means there is an implementation error above
	assert(true && "This should never be reached (node failed to walk to next)");
	return NULL;
}

template <typename storageType>
bool Markov::Node<storageType>::UpdateEdges(Markov::Edge<storageType>* v) {
	this->edges.insert({ v->right()->value(), v });
	//this->total_edge_weights += v->weight();
	return v->traverse();
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::findEdge(storageType repr) {
	return this->edges.find(repr)->second;
};

template <typename storageType>
void Markov::Node<storageType>::updateTotalVerticeWeight(long int offset) {
	this->total_edge_weights += offset;
}

template <typename storageType>
std::map<storageType, Markov::Edge<storageType>*>* Markov::Node<storageType>::Edges() {
	return &(this->edges);
}

template <typename storageType>
uint64_t Markov::Node<storageType>::TotalEdgeWeights() {
	return this->total_edge_weights;
}


 
//new line for the code covarage