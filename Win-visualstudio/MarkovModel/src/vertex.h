#pragma once
#include <cstdint>


namespace Markov {

	class Node;

	class Vertex {
	public:
		
		Markov::Vertex(Markov::Node* _left, Markov::Node* _right);
		
		//adjust weight with the offset value
		void adjust(uint64_t offset);
		
		//return right
		Markov::Node* traverse();

		//setters
		void set_left (Markov::Node*);
		void set_right(Markov::Node*);
		
		//getters
		uint64_t weight();
		Markov::Node* left();
		Markov::Node* right();

	private:
		Markov::Node* _left; //source node
		Markov::Node* _right;//target node
		uint64_t _weight;    //vertex weight
	};


};