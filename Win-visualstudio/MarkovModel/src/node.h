#pragma once
#include <vector>
#include <map>
#include "vertex.h"

namespace Markov {
	class Node {
	public:

		/*
		*  value   => _value
		*  total_vertice_weights  => 0
		*  vertices => []
		*/
		Markov::Node(unsigned char _value);

		/* Link another Markov::Node with this one, 
		*  push it to vertice vector 
		*  and return heap pointer to new vertice
		*
		*  Vertex::left   => this
		*  Vertex::right  => target
		*  Vertex::weight => 0
		*/
		Markov::Vertex* Link(Markov::Node*);
		
		/* Link another Markov::Node from an existing vertex' right.
		*  
		*  return heap pointer of the vertex
		* 
		*  Vertex::left   => this
		*  Vertex::right  => unchanged
		*  Vertex::weight => unchanged
		*/
		Markov::Vertex* Link(Markov::Vertex*);

		/* Select a random vertice based on vertice weights and walk to its Vertex::right.
		*  Return heap pointer to Vertex::right
		*/
		Markov::Node* RandomNext(Markov::Vertex*);

		/* Update the vertice vector. 
		*  Update the total_vertice_weights
		*  Skip if vertice is already in the vector
		*  Return False if duplicate found, true if successful.
		*  
		*  If this is a terminator node, return NULL
		*/
		bool UpdateVertices(Markov::Vertex*);
		
		/* Check if vertice is in the vector.
		*  Return NULL if not found
		*/
		Markov::Vertex* findVertice(Markov::Node* l, Markov::Node* r);

	private:

		//letter held by the node. 255 for start node, 0 for end.
		unsigned char value; 

		//Total weights of the vertices, required by RandomNext;
		uint64_t total_vertice_weights;

		/* Map left is the vertex::right so target can be found with low cost when training.
		*  Makes searching by value cheaper.
		*/
		std::map<unsigned char, Markov::Vertex*> vertices;
	};
};
