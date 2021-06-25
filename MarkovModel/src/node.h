/** @file node.h
 * @brief Node class template
 * @authors Ata Hakçıl, Osman Ömer Yıldıztugay
 * 
 * @copydoc Markov::Node
 */


#pragma once
#include <vector>
#include <map>
#include <assert.h>
#include <iostream>
#include <stdexcept> // To use runtime_error
#include "edge.h"
#include "random.h"
namespace Markov {

	/** @brief A node class that for the vertices of model. Connected with eachother using Edge
	* 
	* This class will later be templated to accept other data types than char*.
	*/
	template <typename storageType>
	class Node {
	public:

		/** @brief Default constructor. Creates an empty Node.
		*/
		Node<storageType>();
		
		/** @brief Constructor. Creates a Node with no edges and with given NodeValue.
		 * @param _value - Nodes character representation.
		 * 
		 * @b Example @b Use: Construct nodes
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
		 * Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
		 * @endcode
		*/
		Node<storageType>(storageType _value);

		/** @brief Link this node with another, with this node as its source.
		 * 
		 * Creates a new Edge.
		 * @param target - Target node which will be the RightNode() of new edge.
		 * @return A new node with LeftNode as this, and RightNode as parameter target.
		 * 
		 * @b Example @b Use: Construct nodes
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
		 * Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
		 * Markov::Edge<unsigned char>* e = LeftNode->Link(RightNode);
		 * @endcode
		*/
		Edge<storageType>* Link(Node<storageType>*);
		
		/** @brief Link this node with another, with this node as its source.
		 * 
		 * *DOES NOT* create a new Edge.
		 * @param Edge - Edge that will accept this node as its LeftNode.
		 * @return the same edge as parameter target.
		 * 
		 * @b Example @b Use: Construct and link nodes
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
		 * Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
		 * Markov::Edge<unsigned char>* e = LeftNode->Link(RightNode);
		 * LeftNode->Link(e);
		 * @endcode
		*/
		Edge<storageType>* Link(Edge<storageType>*);

		/** @brief Chose a random node from the list of edges, with regards to its EdgeWeight, and TraverseNode to that.
		 * 
		 * This operation is done by generating a random number in range of 0-this.total_edge_weights, and then iterating over the list of edges.
		 * At each step, EdgeWeight of the edge is subtracted from the random number, and once it is 0, next node is selected.
		 * @return Node that was chosen at EdgeWeight biased random.
		 * 
		 * @b Example @b Use: Use randomNext to do a random walk on the model
		 * @code{.cpp} 
		 *  char* buffer[64];
		 *  Markov::Model<char> model;
		 *  model.Import("model.mdl");
		 * 	Markov::Node<char>* n = model.starterNode;
		 *	int len = 0;
		 *	Markov::Node<char>* temp_node;
		 *	while (true) {
		 *		temp_node = n->RandomNext(randomEngine);
		 *		if (len >= maxSetting) {
		 *			break;
		 *		}
		 *		else if ((temp_node == NULL) && (len < minSetting)) {
		 *			continue;
		 *		}
		 *
		 *		else if (temp_node == NULL){
		 *			break;
		 *		}
		 *			
		 *		n = temp_node;
		 *
		 *		buffer[len++] = n->NodeValue();
		 *	}
		 * @endcode
		*/
		Node<storageType>* RandomNext(Markov::Random::RandomEngine* randomEngine);

		/** @brief Insert a new edge to the this.edges.
		 * @param edge - New edge that will be inserted.
		 * @return true if insertion was successful, false if it fails.
         * 
		 * @b Example @b Use: Construct and update edges
		 * 
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
		 * Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
		 * Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
		 * Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
		 * Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>(src, target2);
		 * e1->AdjustEdge(25);
		 * src->UpdateEdges(e1);
		 * e2->AdjustEdge(30);
		 * src->UpdateEdges(e2);
		 * @endcode
		*/
		bool UpdateEdges(Edge<storageType>*);
		
		/** @brief Find an edge with its character representation.
		 * @param repr - character NodeValue of the target node.
		 * @return Edge that is connected between this node, and the target node.
		 *  
		 * @b Example @b Use: Construct and update edges
		 * 
		 * @code{.cpp}
		 * Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
		 * Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
		 * Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
		 * Markov::Edge<unsigned char>* res = NULL;
		 * src->Link(target1);
		 * src->Link(target2);			
 		 * res = src->FindEdge('b');
		 *  
		 * @endcode
		 *  
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
		inline unsigned char NodeValue();

		/** @brief Change total weights with offset
		 * @param offset to adjust the vertice weight with
		*/
		void UpdateTotalVerticeWeight(long int offset);

		/** @brief return edges
		*/
		inline std::map<storageType, Edge<storageType>*>* Edges();

		/** @brief return total edge weights
		*/
		inline long int TotalEdgeWeights();


		std::vector<Edge<storageType>*> edgesV;
	private:

		/** 
			@brief Character representation of this node. 0 for starter, 0xff for terminator.
		*/
		storageType _value; 

		/** 
			@brief Total weights of the vertices, required by RandomNext
		*/
		long int total_edge_weights;

		/** 
			@brief A map of all edges connected to this node, where this node is at the LeftNode.
			* Map is indexed by unsigned char, which is the character representation of the node.
		*/
		std::map<storageType, Edge<storageType>*> edges;
	};
};









template <typename storageType>
Markov::Node<storageType>::Node(storageType _value) {
	this->_value = _value;
	this->total_edge_weights = 0L;
};

template <typename storageType>
Markov::Node<storageType>::Node() {
	this->_value = 0;
	this->total_edge_weights = 0L;
};

template <typename storageType>
inline unsigned char Markov::Node<storageType>::NodeValue() {
	return _value;
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::Link(Markov::Node<storageType>* n) {
	Markov::Edge<storageType>* v = new Markov::Edge<storageType>(this, n);
	this->UpdateEdges(v);
	return v;
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::Link(Markov::Edge<storageType>* v) {
	v->SetLeftEdge(this);
	this->UpdateEdges(v);
	return v;
}

template <typename storageType>
Markov::Node<storageType>* Markov::Node<storageType>::RandomNext(Markov::Random::RandomEngine* randomEngine) {

	//get a random NodeValue in range of total_vertice_weight
	long int selection = randomEngine->random() % this->total_edge_weights;//distribution()(generator());// distribution(generator);
	//make absolute, no negative modulus values wanted
	//selection = (selection >= 0) ? selection : (selection + this->total_edge_weights);
	for(int i=0;i<this->edgesV.size();i++){
		selection -= this->edgesV[i]->EdgeWeight();
		if (selection < 0) return this->edgesV[i]->TraverseNode();
	}

	//if this assertion is reached, it means there is an implementation error above
	std::cout << "This should never be reached (node failed to walk to next)\n"; //cant assert from child thread
	assert(true && "This should never be reached (node failed to walk to next)");
	return NULL;
}

template <typename storageType>
bool Markov::Node<storageType>::UpdateEdges(Markov::Edge<storageType>* v) {
	this->edges.insert({ v->RightNode()->NodeValue(), v });
	this->edgesV.push_back(v);
	//this->total_edge_weights += v->EdgeWeight();
	return v->TraverseNode();
}

template <typename storageType>
Markov::Edge<storageType>* Markov::Node<storageType>::FindEdge(storageType repr) {
	auto e = this->edges.find(repr);
	if (e == this->edges.end()) return NULL;
	return e->second;
};

template <typename storageType>
void Markov::Node<storageType>::UpdateTotalVerticeWeight(long int offset) {
	this->total_edge_weights += offset;
}

template <typename storageType>
inline std::map<storageType, Markov::Edge<storageType>*>* Markov::Node<storageType>::Edges() {
	return &(this->edges);
}

template <typename storageType>
inline long int Markov::Node<storageType>::TotalEdgeWeights() {
	return this->total_edge_weights;
}

