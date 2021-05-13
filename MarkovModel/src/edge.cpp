#include "edge.h"
#include <cstddef>

//default constructor of edge
template <typename NodeStorageType>
Markov::Edge<NodeStorageType>::Edge() {
    this->_left = NULL;
    this->_right = NULL;
    this->_weight = 0;
}
//constructor of edge
template <typename NodeStorageType>
Markov::Edge<NodeStorageType>::Edge(Markov::Node<NodeStorageType>* _left, Markov::Node<NodeStorageType>* _right) {
    this->_left = _left;
    this->_right = _right;
    this->_weight = 0;
}
//to adjust the edges by the edge with its offset
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::adjust(uint64_t offset) {
    this->_weight += offset;
    this->left()->updateTotalVerticeWeight(offset);
}
//to traverse the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::traverse() {
    if (this->right()->value() == 0xff) //terminator node
        return NULL;
    return _right;
}
//to set the left of the node
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::set_left(Markov::Node<NodeStorageType>* n) {
    this->_left = n;
}
//to set the right of the node
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::set_right(Markov::Node<NodeStorageType>* n) {
    this->_right = n;
}
//to get the weight of the node
template <typename NodeStorageType>
uint64_t Markov::Edge<NodeStorageType>::weight() {
    return this->_weight;
}
//to get the left of the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::left() {
    return this->_left;
}
//to get the right of the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::right() {
    return this->_right;
}

