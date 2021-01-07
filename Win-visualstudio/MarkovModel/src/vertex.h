#pragma once
#include <cstdint>


namespace Markov {

	class Node;

	class Vertex {
	public:
		
		Markov::Vertex(Markov::Node* _left, Markov::Node* _right) {
			this->left = _left;
			this->right = _right;
			this->weight = 0;
		}
		
		//adjust weight with the offset value
		void adjust(uint64_t offset) {
			this->weight += offset;
		}
		
		//return right
		Markov::Node* traverse() {
			return left;
		}

	private:
		Markov::Node* left; //source node
		Markov::Node* right;//target node
		uint64_t weight;    //vertex weight
	};


};