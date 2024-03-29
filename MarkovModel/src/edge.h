/** @file edge.h
 * @brief Edge class template
 * @authors Ata Hakçıl, Osman Ömer Yıldıztugay
 * 
 * @copydoc Markov::Edge
 */

#pragma once
#include <cstdint>
#include <cstddef>

namespace Markov {

	template <typename NodeStorageType>
	class Node;

	/** @brief Edge class used to link nodes in the model together.
	* 
	Has LeftNode, RightNode, and EdgeWeight of the edge.
	Edges are *UNIDIRECTIONAL* in this model. They can only be traversed LeftNode to RightNode.
	*/
	template <typename NodeStorageType>
	class Edge {
	public:
		
		/** @brief Default constructor. 
 		*/
		Edge<NodeStorageType>();

		/** @brief Constructor. Initialize edge with given RightNode and LeftNode
		 * @param _left - Left node of this edge.
		 * @param _right - Right node of this edge.
		 * 
		 * @b Example @b Use: Construct edge
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
		 * Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
		 * Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
		 * @endcode
		 *
		*/
		Edge<NodeStorageType>(Node<NodeStorageType>* _left, Node<NodeStorageType>* _right);
		
		/** @brief Adjust the edge EdgeWeight with offset.
		 * Adds the offset parameter to the edge EdgeWeight.
		 * @param offset - NodeValue to be added to the EdgeWeight
		 * 
		 * @b Example @b Use: Construct edge
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
		 * Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
		 * Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
		 * 
		 * e1->AdjustEdge(25);
		 * 
		 * @endcode
		*/
		void AdjustEdge(long int offset);
		
		/** @brief Traverse this edge to RightNode.
		 * @return Right node. If this is a terminator node, return NULL
		 * 
		 * 
		 * @b Example @b Use: Traverse a node
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
		 * Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
		 * Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
		 * 
		 * e1->AdjustEdge(25);
		 * Markov::Edge<unsigned char>* e2 = e1->traverseNode();
		 * @endcode
		 * 
		*/
		inline Node<NodeStorageType>* TraverseNode();

		/** @brief Set LeftNode of this edge.
		* @param node - Node to be linked with.
		*/
		void SetLeftEdge (Node<NodeStorageType>*);
		/** @brief Set RightNode of this edge.
		* @param node - Node to be linked with.
		*/
		void SetRightEdge(Node<NodeStorageType>*);
		
		/** @brief return edge's EdgeWeight.
		* @return edge's EdgeWeight.
		*/
		inline uint64_t EdgeWeight();

		/** @brief return edge's LeftNode
		* @return edge's LeftNode.
		*/
		Node<NodeStorageType>* LeftNode();

		/** @brief return edge's RightNode
		* @return edge's RightNode.
		*/
		inline Node<NodeStorageType>* RightNode();

	private:
		/** 
			@brief source node 
		*/
		Node<NodeStorageType>* _left; 

		/** 
			@brief target node 
		*/
		Node<NodeStorageType>* _right;

		/** @brief
			Edge Edge Weight 
		*/
		long int _weight;
	};


};

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
void Markov::Edge<NodeStorageType>::AdjustEdge(long int offset) {
	this->_weight += offset;
	this->LeftNode()->UpdateTotalVerticeWeight(offset);
}
//to TraverseNode the node
template <typename NodeStorageType>
inline Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::TraverseNode() {
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
inline uint64_t Markov::Edge<NodeStorageType>::EdgeWeight() {
	return this->_weight;
}
//to get the LeftNode of the node
template <typename NodeStorageType>
Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::LeftNode() {
	return this->_left;
}
//to get the RightNode of the node
template <typename NodeStorageType>
inline Markov::Node<NodeStorageType>* Markov::Edge<NodeStorageType>::RightNode() {
	return this->_right;
}


