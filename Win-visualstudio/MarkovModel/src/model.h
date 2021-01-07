#pragma once
#include <map>
#include <vector>
#include <fstream>

#include "vertex.h"
#include "node.h"

namespace Markov {
	
	class Model {
	public:
		/* Traverse the model by calling Markov::Node::RandomNext on the start node 
		*  and repeating until terminator node. (will return NULL)

				var dataString ="We assume we are getting something from our dataset";
		var order = 3;
		var ngrams = {};



		for (int i = 0;i <= s.length - order;i++) {  // its for 3-gram occurance
			var gram = dataString.substring(i, i + 3);
			ngrams.push(gram);
			if (!ngrams[gram]) {
				ngrams[gram] = []; // when i see 2gram what comes after to see in array
				ngrams[gram].push(dataString.charAt(i + 3));
			}
			ngrams[gram].push(s.charAt(i + 3));
		}
		function markoving() {
			var currentGram = dataString.substring(0, order);
			var possibilities = ngrams[curremtGram];
			var nextThing = radom(possibilities); // to give random elemnts from array
			var result = currentGram + nextThing;

		}

		*/
		void RandomWalk();

		/* Adjust the model with a single string
		*  Traverse string char by char and adjust each vertice with occurrence.
		*  Param is signed so negative bias can be applied.
		*/
		void adjust(char* string, long int occurrence);

		/* Import model structure from a savefile
		*/
		bool Import(std::ifstream);
		bool Import(char* filename);

		/* Export model structure to a savefile
		*/
		bool Export(std::ofstream);
		bool Export(char* filename);

	private:
		/* Map left is the Nodes value
		* Map right is the node pointer
		*/
		std::map<unsigned char, Markov::Node*> nodes;

		// A list of all vertices
		// Might drop this in implementation, no use so far
		std::vector<Markov::Vertex> vertices;
	};

};