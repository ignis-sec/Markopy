#include "node.h"
#include <assert.h>

/*
*  value   => _value
*  total_vertice_weights  => 0
*  vertices => []
*/
Markov::Node::Node(unsigned char _value) {
	this->_value = _value;
};

//empty initializer
Markov::Node::Node() {
	this->_value = 0;
};

unsigned char Markov::Node::value() {
	return _value;
}

/* Link another Markov::Node with this one,
*  push it to vertice vector
*  and return heap pointer to new vertice
*
*  Vertex::left   => this
*  Vertex::right  => target
*  Vertex::weight => 0
*/
Markov::Vertex* Markov::Node::Link(Markov::Node* n) {
	Markov::Vertex *v = new Markov::Vertex(this, n);
	this->UpdateVertices(v);
	return v;
}

/* Link another Markov::Node from an existing vertex' right.
*
*  return heap pointer of the vertex
*
*  Vertex::left   => this
*  Vertex::right  => unchanged
*  Vertex::weight => unchanged
*/
Markov::Vertex* Markov::Node::Link(Markov::Vertex* v) {
	v->set_left(this);
	this->UpdateVertices(v);
	return v;
}

/* Select a random vertice based on vertice weights and walk to its Vertex::right.
*  Return heap pointer to Vertex::right
*/
Markov::Node* Markov::Node::RandomNext() {

	//get a random value in range of total_vertice_weight
	int selection = rand() % this->total_vertice_weights;
	
	//make absolute, no negative modulus values wanted
	selection = (selection<0)? selection : selection + this->total_vertice_weights;

	//iterate over the vertex map
	//Subtract the vertex weight from the selection at each vertex
	//when selection goes below 0, pick that node 
	//(Fast random selection with weight bias)
	for ( std::pair<const unsigned char,Markov::Vertex*> const& x : this->vertices) {
		selection -= x.second->weight();
		if (selection < 0) return x.second->traverse();
	}

	//if this assertion is reached, it means there is an implementation error above
	assert(true && "This should never be reached (node failed to walk to next)");

}


/* Update the vertice vector.
*  Update the total_vertice_weights
*  Skip if vertice is already in the vector
*  Return False if duplicate found, true if successful.
*
*  If this is a terminator node, return NULL
*/
bool Markov::Node::UpdateVertices(Markov::Vertex* v) {
	this->vertices.insert({ v->traverse()->value(), v });
	this->total_vertice_weights += v->weight();
	return v->traverse();
}

/* Check if vertice is in the vector.
*  Return NULL if not found
*/
Markov::Vertex* findVertice(Markov::Node* l, Markov::Node* r) {/*TODO*/};

