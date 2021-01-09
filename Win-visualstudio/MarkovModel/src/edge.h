#pragma once
#include <cstdint>


namespace Markov {

	class Node;
	/** @brief Edge class used to link nodes in the model together.
	* 
	Has left, right, and weight of the edge.
	Edges are *UNIDIRECTIONAL* in this model. They can only be traversed left to right.
	*/
	class Edge {
	public:
		
		/** @brief Default constructor. 
 		*/
		Markov::Edge();
		/**@brief Constructor. Initialize edge with given right and left
		* @param _left - Left node of this edge.
		* @param _right - Right node of this edge.
		*/
		Markov::Edge(Markov::Node* _left, Markov::Node* _right);
		
		/** @brief Adjust the edge weight with offset.
		* Adds the offset parameter to the edge weight.
		* @param offset - value to be added to the weight
		*/
		void adjust(uint64_t offset);
		
		/** @brief Traverse this edge to right.
		* @return Right node. If this is a terminator node, return NULL
		*/
		Markov::Node* traverse();

		/** @brief Set left of this edge.
		* @param node - Node to be linked with.
		*/
		void set_left (Markov::Node*);
		/** @brief Set right of this edge.
		* @param node - Node to be linked with.
		*/
		void set_right(Markov::Node*);
		
		/** @brief return edge's weight.
		* @return edge's weight.
		*/
		uint64_t weight();

		/** @brief return edge's left
		* @return edge's left.
		*/
		Markov::Node* left();

		/** @brief return edge's right
		* @return edge's right.
		*/
		Markov::Node* right();

	private:
		Markov::Node* _left; /** @brief source node*/
		Markov::Node* _right;/** @brief target node*/
		uint64_t _weight;    /** @brief Edge weight*/
	};


};