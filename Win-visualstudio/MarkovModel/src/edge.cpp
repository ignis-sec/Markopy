#include "edge.h"
#include "node.h"

//Empty constructor
Markov::Edge::Edge() {
	this->_left = NULL;
	this->_right = NULL;
	this->_weight = 0;
}

//Basic constructor
Markov::Edge::Edge(Markov::Node* _left, Markov::Node* _right) {
	this->_left = _left;
	this->_right = _right;
	this->_weight = 0;
}

//adjust weight with the offset value
void Markov::Edge::adjust(uint64_t offset) {
	this->_weight += offset;
}



//return right
Markov::Node* Markov::Edge::traverse() {
	if (this->right()->value() == 0xff) //terminator node
		return NULL;
	return _left;
}


//Getters and setters below 
void Markov::Edge::set_left(Markov::Node* n) {
	this->_left = n;
}

void Markov::Edge::set_right(Markov::Node* n) {
	this->_right = n;
}

uint64_t Markov::Edge::weight() {
	return this->_weight;
}

Markov::Node* Markov::Edge::left() {
	return this->_left;
}

Markov::Node* Markov::Edge::right() {
	return this->_right;
}