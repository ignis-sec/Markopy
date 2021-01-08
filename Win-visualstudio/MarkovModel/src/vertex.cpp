#include "vertex.h"

class Markov::Node;


Markov::Vertex::Vertex() {
	this->_left = NULL;
	this->_right = NULL;
	this->_weight = 0;
}

Markov::Vertex::Vertex(Markov::Node* _left, Markov::Node* _right) {
	this->_left = _left;
	this->_right = _right;
	this->_weight = 0;
}

//adjust weight with the offset value
void Markov::Vertex::adjust(uint64_t offset) {
	this->_weight += offset;
}



//return right
Markov::Node* Markov::Vertex::traverse() {
	return _left;
}

void Markov::Vertex::set_left(Markov::Node* n) {
	this->_left = n;
}

void Markov::Vertex::set_right(Markov::Node* n) {
	this->_right = n;
}

uint64_t Markov::Vertex::weight() {
	return this->_weight;
}

Markov::Node* Markov::Vertex::left() {
	return this->_left;
}

Markov::Node* Markov::Vertex::right() {
	return this->_right;
}