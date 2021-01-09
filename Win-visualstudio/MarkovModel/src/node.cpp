#include "node.h"
#include <assert.h>

Markov::Node::Node(unsigned char _value) {
	this->_value = _value;
};

Markov::Node::Node() {
	this->_value = 0;
};

unsigned char Markov::Node::value() {
	return _value;
}

Markov::Edge* Markov::Node::Link(Markov::Node* n) {
	Markov::Edge *v = new Markov::Edge(this, n);
	this->UpdateEdges(v);
	return v;
}

Markov::Edge* Markov::Node::Link(Markov::Edge* v) {
	v->set_left(this);
	this->UpdateEdges(v);
	return v;
}

Markov::Node* Markov::Node::RandomNext() {

	//get a random value in range of total_vertice_weight
	int selection = rand() % this->total_edge_weights;
	
	//make absolute, no negative modulus values wanted
	selection = (selection<0)? selection : selection + this->total_edge_weights;

	//iterate over the Edge map
	//Subtract the Edge weight from the selection at each Edge
	//when selection goes below 0, pick that node 
	//(Fast random selection with weight bias)
	for ( std::pair<const unsigned char,Markov::Edge*> const& x : this->edges) {
		selection -= x.second->weight();
		if (selection < 0) return x.second->traverse();
	}

	//if this assertion is reached, it means there is an implementation error above
	assert(true && "This should never be reached (node failed to walk to next)");

}

bool Markov::Node::UpdateEdges(Markov::Edge* v) {
	this->edges.insert({ v->traverse()->value(), v });
	this->total_edge_weights += v->weight();
	return v->traverse();
}

Markov::Edge* findVertice(Markov::Node* l, Markov::Node* r) {/*TODO*/};

