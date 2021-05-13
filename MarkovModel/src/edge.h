#pragma once
#include <cstdint>


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
		/**@brief Constructor. Initialize edge with given RightNode and LeftNode
		* @param _left - Left node of this edge.
		* @param _right - Right node of this edge.
		*/
		
		Edge<NodeStorageType>(Node<NodeStorageType>* _left, Node<NodeStorageType>* _right);
		
		/** @brief Adjust the edge EdgeWeight with offset.
		* Adds the offset parameter to the edge EdgeWeight.
		* @param offset - NodeValue to be added to the EdgeWeight
		*/
		void AdjustEdge(uint64_t offset);
		
		/** @brief Traverse this edge to RightNode.
		* @return Right node. If this is a terminator node, return NULL
		*/
		Node<NodeStorageType>* TraverseNode();

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
		uint64_t EdgeWeight();

		/** @brief return edge's LeftNode
		* @return edge's LeftNode.
		*/
		Node<NodeStorageType>* LeftNode();

		/** @brief return edge's RightNode
		* @return edge's RightNode.
		*/
		Node<NodeStorageType>* RightNode();

	private:
		Node<NodeStorageType>* _left; /** @brief source node*/
		Node<NodeStorageType>* _right;/** @brief target node*/
		int _weight;    /** @brief Edge EdgeWeight*/
	};


};
//new line for the code covarage