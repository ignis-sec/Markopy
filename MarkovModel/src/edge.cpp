#include "edge.h"
#include <cstddef>

template <typename NodeStorageType>
Markov::Edge<NodeStorageType>::Edge() {
	this->_left = NULL;
	this->_right = NULL;
	this->_weight = 0;
}

template <typename NodeStorageType>
Markov::Edge<NodeStorageType>::Edge(Markov::Node<NodeStorageType>* _left, Markov::Node<NodeStorageType>* _right) {
	this->_left = _left;
	this->_right = _right;
	this->_weight = 0;
}

template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::adjust(uint64_t offset) {
	this->_weight += offset;
	this->left()->updateTotalVerticeWeight(offset);
}

template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::traverse() {
	if (this->right()->value() == 0xff) //terminator node
		return NULL;
	return _right;
}

template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::set_left(Markov::Node<NodeStorageType>* n) {
	this->_left = n;
}

template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::set_right(Markov::Node<NodeStorageType>* n) {
	this->_right = n;
}

template <typename NodeStorageType>
uint64_t Markov::Edge<NodeStorageType>::weight() {
	return this->_weight;
}

template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::left() {
	return this->_left;
}

template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::right() {
	return this->_right;
}