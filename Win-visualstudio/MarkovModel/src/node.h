#pragma once
#include <vector>
#include <map>
#include "edge.h"

namespace Markov {

	/** @brief A node class that for the vertices of model. Connected with eachother using Markov::Edge
	* 
	* This class will *later be templated to accept other data types than char*.
	*/
	class Node {
	public:

		/** @brief Default constructor. Creates an empty Node.
		*/
		Markov::Node();
		/** @brief Constructor. Creates a Node with no edges and with given value.
		* @param value - Nodes character representation.
		*/
		Markov::Node(unsigned char _value);

		/** @brief Link this node with another, with this node as its source.
		* 
		* Creates a new Edge.
		* @param target - Target node which will be the right() of new edge.
		* @return A new node with left as this, and right as parameter target.
		*/
		Markov::Edge* Link(Markov::Node*);
		
		/** @brief Link this node with another, with this node as its source.
		* 
		* *DOES NOT* create a new Edge.
		* @param Edge - Edge that will accept this node as its left.
		* @return the same edge as parameter target.
		*/
		Markov::Edge* Link(Markov::Edge*);

		/** @brief Chose a random node from the list of edges, with regards to its weight, and traverse to that.
		* 
		* This operation is done by generating a random number in range of 0-this.total_edge_weights, and then iterating over the list of edges.
		* At each step, weight of the edge is subtracted from the random number, and once it is 0, next node is selected.
		* @return Node that was chosen at weight biased random.
		*/
		Markov::Node* RandomNext();

		/** @brief Insert a new edge to the this.edges.
		* @param edge - New edge that will be inserted.
		* @return true if insertion was successful, false if it fails.
		*/
		bool UpdateEdges(Markov::Edge*);
		
		/** @brief Find an edge with its character representation.
		* @param repr - character value of the target node.
		* @return Edge that is connected between this node, and the target node.
		*/
		Markov::Edge* findEdge(char repr);

		/** @brief Find an edge with its pointer. Avoid unless neccessary because comptutational cost of find by character is cheaper (because of std::map)
		* @param target - target node.
		* @return Edge that is connected between this node, and the target node.
		*/
		Markov::Edge* findEdge(Node* target);
		
		/** @brief Return character representation of this node.
		* @return character representation at _value.
		*/
		unsigned char value();

	private:

		
		unsigned char _value; /** @brief Character representation of this node. 0 for starter, 0xff for terminator.*/

		uint64_t total_edge_weights;/** @brief Total weights of the vertices, required by RandomNext;*/

		/** @brief A map of all edges connected to this node, where this node is at the left.
		* 
		* Map is indexed by unsigned char, which is the character representation of the node.
		*/
		std::map<unsigned char, Markov::Edge*> edges;
	};
};
