#pragma once
#include <vector>
#include <map>
#include "edge.h"

namespace Markov {
	class Node {
	public:

		/*
		*  value   => _value
		*  total_vertice_weights  => 0
		*  vertices => []
		*/
		Markov::Node();
		Markov::Node(unsigned char _value);

		/* Link another Markov::Node with this one, 
		*  push it to vertice vector 
		*  and return heap pointer to new vertice
		*
		*  Edge::left   => this
		*  Edge::right  => target
		*  Edge::weight => 0
		*/
		Markov::Edge* Link(Markov::Node*);
		
		/* Link another Markov::Node from an existing Edge' right.
		*  
		*  return heap pointer of the Edge
		* 
		*  Edge::left   => this
		*  Edge::right  => unchanged
		*  Edge::weight => unchanged
		*/
		Markov::Edge* Link(Markov::Edge*);

		/* Select a random vertice based on vertice weights and walk to its Edge::right.
		*  Return heap pointer to Edge::right
		*/
		Markov::Node* RandomNext();

		/* Update the vertice vector. 
		*  Update the total_vertice_weights
		*  Skip if vertice is already in the vector
		*  Return False if duplicate found, true if successful.
		*  
		*  If this is a terminator node, return NULL
		*/
		bool UpdateEdges(Markov::Edge*);
		
		/* Check if vertice is in the vector.
		*  Return NULL if not found
		*/
		Markov::Edge* findEdge(Markov::Node* l, Markov::Node* r);
		
		unsigned char value();

	private:

		//letter held by the node. 255 for start node, 0 for end.
		unsigned char _value; 

		//Total weights of the vertices, required by RandomNext;
		uint64_t total_edge_weights;

		/* Map left is the Edge::right so target can be found with low cost when training.
		*  Makes searching by value cheaper.
		*/
		std::map<unsigned char, Markov::Edge*> edges;
	};
};
