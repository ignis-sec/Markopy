#include "vertex.h"

class Markov::Node;

Markov::Vertex::Vertex(Markov::Node* _left, Markov::Node* _right) {
	this->left = _left;
	this->right = _right;
	this->weight = 0;
}

//adjust weight with the offset value
void Markov::Vertex::adjust(uint64_t offset) {
	this->weight += offset;
}

//return right
Markov::Node* Markov::Vertex::traverse() {
	return left;
}