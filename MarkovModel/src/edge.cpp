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
//to AdjustEdge the edges by the edge with its offset
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::AdjustEdge(uint64_t offset) {
    this->_weight += offset;
    this->LeftNode()->UpdateTotalVerticeWeight(offset);
}
//to TraverseNode the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::TraverseNode() {
    if (this->RightNode()->NodeValue() == 0xff) //terminator node
        return NULL;
    return _right;
}
//to set the LeftNode of the node
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::SetLeftEdge(Markov::Node<NodeStorageType>* n) {
    this->_left = n;
}
//to set the RightNode of the node
template <typename NodeStorageType>
void Markov::Edge<NodeStorageType>::SetRightEdge(Markov::Node<NodeStorageType>* n) {
    this->_right = n;
}
//to get the EdgeWeight of the node
template <typename NodeStorageType>
uint64_t Markov::Edge<NodeStorageType>::EdgeWeight() {
    return this->_weight;
}
//to get the LeftNode of the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::LeftNode() {
    return this->_left;
}
//to get the RightNode of the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::RightNode() {
    return this->_right;
}


