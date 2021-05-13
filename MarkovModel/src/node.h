#pragma once
#include <vector>
#include <map>
#include "edge.h"
#include <random>

extern std::random_device rd;
extern std::default_random_engine generator;
extern std::uniform_int_distribution<long long unsigned> distribution;

namespace Markov {

	/** @brief A node class that for the vertices of model. Connected with eachother using Edge
	* 
	* This class will *later be templated to accept other data types than char*.
	*/
	template <typename storageType>
	class Node {
	public:

		/** @brief Default constructor. Creates an empty Node.
		*/
		Node<storageType>();
		
		/** @brief Constructor. Creates a Node with no edges and with given NodeValue.
		* @param NodeValue - Nodes character representation.
		*/
		Node<storageType>(storageType _value);

		/** @brief Link this node with another, with this node as its source.
		* 
		* Creates a new Edge.
		* @param target - Target node which will be the RightNode() of new edge.
		* @return A new node with LeftNode as this, and RightNode as parameter target.
		*/
		Edge<storageType>* Link(Node<storageType>*);
		
		/** @brief Link this node with another, with this node as its source.
		* 
		* *DOES NOT* create a new Edge.
		* @param Edge - Edge that will accept this node as its LeftNode.
		* @return the same edge as parameter target.
		*/
		Edge<storageType>* Link(Edge<storageType>*);

		/** @brief Chose a random node from the list of edges, with regards to its EdgeWeight, and TraverseNode to that.
		* 
		* This operation is done by generating a random number in range of 0-this.total_edge_weights, and then iterating over the list of edges.
		* At each step, EdgeWeight of the edge is subtracted from the random number, and once it is 0, next node is selected.
		* @return Node that was chosen at EdgeWeight biased random.
		*/
		Node<storageType>* RandomNext();

		/** @brief Insert a new edge to the this.edges.
		* @param edge - New edge that will be inserted.
		* @return true if insertion was successful, false if it fails.
		*/
		bool UpdateEdges(Edge<storageType>*);
		
		/** @brief Find an edge with its character representation.
		* @param repr - character NodeValue of the target node.
		* @return Edge that is connected between this node, and the target node.
		*/
		Edge<storageType>* FindEdge(storageType repr);

		/** @brief Find an edge with its pointer. Avoid unless neccessary because comptutational cost of find by character is cheaper (because of std::map)
		* @param target - target node.
		* @return Edge that is connected between this node, and the target node.
		*/
		Edge<storageType>* FindEdge(Node<storageType>* target);
		
		/** @brief Return character representation of this node.
		* @return character representation at _value.
		*/
		unsigned char NodeValue();

		/** @brief Change total weights with offset
		*/
		void UpdateTotalVerticeWeight(long int offset);

		/** @brief return edges
		*/
		std::map<storageType, Edge<storageType>*>* Edges();

		/** @brief return total edge weights
		*/
		uint64_t TotalEdgeWeights();


	private:

		
		storageType _value; /** @brief Character representation of this node. 0 for starter, 0xff for terminator.*/

		int total_edge_weights;/** @brief Total weights of the vertices, required by RandomNext;*/

		/** @brief A map of all edges connected to this node, where this node is at the LeftNode.
		* 
		* Map is indexed by unsigned char, which is the character representation of the node.
		*/
		std::map<storageType, Edge<storageType>*> edges;
	};
};
//new line for the code covarage