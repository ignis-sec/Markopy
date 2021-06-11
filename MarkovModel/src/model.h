/** @dir Model.h
 *
 */


#pragma once
#include <map>
#include <vector>
#include <fstream>
#include <assert.h>
#include <string>
#include <algorithm>
#include "node.h"
#include "edge.h"

/**
	@brief Namespace for the markov-model related classes. 
	Contains Model, Node and Edge classes
*/
namespace Markov {

	template <typename NodeStorageType>
	class Node;

	template <typename NodeStorageType>
	class Edge;

	template <typename NodeStorageType>

	/** @brief class for the final Markov Model, constructed from nodes and edges.
	 * 
	 * Each atomic piece of the generation result is stored in a node, while edges contain the relation weights.
	 * *Extending:*
	 * To extend the class, implement the template and inherit from it, as "class MyModel : public Markov::Model<char>". 
	 * For a complete demonstration of how to extend the class, see MarkovPasswords.
	 *
	 * Whole model can be defined as a list of the edges, as dangling nodes are pointless. This approach is used for the import/export operations.
	 * For more information on importing/exporting model, check out the github readme and wiki page.
	 * 
	*/
	class Model {
	public:
		
		/** @brief Initialize a model with only start and end nodes.
		 * 
		 * Initialize an empty model with only a starterNode
		 * Starter node is a special kind of node that has constant 0x00 value, and will be used to initiate the generation execution from.
		*/
		Model<NodeStorageType>();
			
		/** @brief Do a random walk on this model. 
		 * 
		 * Start from the starter node, on each node, invoke RandomNext using the random engine on current node, until terminator node is reached.
		 * If terminator node is reached before minimum length criateria is reached, ignore the last selection and re-invoke randomNext
		 * 
		 * If maximum length criteria is reached but final node is not, cut off the generation and proceed to the final node.
		 * This function takes Markov::Random::RandomEngine as a parameter to generate pseudo random numbers from
		 * 
		 * This library is shipped with two random engines, Marsaglia and Mersenne. While mersenne output is higher in entropy, most use cases
		 * don't really need super high entropy output, so Markov::Random::Marsaglia is preferable for better performance.
		 * 
		 * This function WILL NOT reallocate buffer. Make sure no out of bound writes are happening via maximum length criteria.
		 * 
		 * @b Example @b Use: Generate 10 lines, with 5 to 10 characters, and print the output. Use Marsaglia
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * Model.import("model.mdl");
		 * char* res = new char[11];
		 * Markov::Random::Marsaglia MarsagliaRandomEngine;
		 * for (int i = 0; i < 10; i++) {
		 *		this->RandomWalk(&MarsagliaRandomEngine, 5, 10, res); 
		 *		std::cout << res << "\n";
		 *	}
		 * @endcode
		 * 
		 * @param randomEngine Random Engine to use for the random walks. For examples, see Markov::Random::Mersenne and Markov::Random::Marsaglia
		 * @param minSetting Minimum number of characters to generate
		 * @param maxSetting Maximum number of character to generate
		 * @param buffer buffer to write the result to
		 * @return Null terminated string that was generated.
		*/
		NodeStorageType* RandomWalk(Markov::Random::RandomEngine* randomEngine, int minSetting, int maxSetting, NodeStorageType* buffer);

		/** @brief Adjust the model with a single string. 
		 * 
		 * Start from the starter node, and for each character, AdjustEdge the edge EdgeWeight from current node to the next, until NULL character is reached.
		 * 
		 * Then, update the edge EdgeWeight from current node, to the terminator node.
		 * 
		 * This function is used for training purposes, as it can be used for adjusting the model with each line of the corpus file.
		 * 
		 * @b Example @b Use: Create an empty model and train it with string: "testdata"
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * char test[] = "testdata";
		 * model.AdjustEdge(test, 15); 
		 * @endcode
		 * 
		 * 
		 * @param string - String that is passed from the training, and will be used to AdjustEdge the model with
		 * @param occurrence - Occurrence of this string. 
		 * 
		 * 
		*/
		void AdjustEdge(const NodeStorageType* payload, long int occurrence);

		/** @brief Import a file to construct the model. 
		 * 
	 	 * File contains a list of edges. For more info on the file format, check out the wiki and github readme pages.
		 * Format is: Left_repr;EdgeWeight;right_repr
		 * 
		 * Iterate over this list, and construct nodes and edges accordingly. 
		 * @return True if successful, False for incomplete models or corrupt file formats
		 * 
		 * @b Example @b Use: Import a file from ifstream
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * std::ifstream file("test.mdl");
		 * model.Import(&file);
		 * @endcode
		*/
		bool Import(std::ifstream*);

		/** @brief Open a file to import with filename, and call bool Model::Import with std::ifstream
		 * @return True if successful, False for incomplete models or corrupt file formats
		 * 
		 * @b Example @b Use: Import a file with filename
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * model.Import("test.mdl");
		 * @endcode
		*/
		bool Import(const char* filename);

		/** @brief Export a file of the model.
		 *
		 * File contains a list of edges.
		 * Format is: Left_repr;EdgeWeight;right_repr.
		 * For more information on the format, check out the project wiki or github readme.
		 * 
		 * Iterate over this vertices, and their edges, and write them to file.
		 * @return True if successful, False for incomplete models.
		 * 
		 * @b Example @b Use: Export file to ofstream
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * std::ofstream file("test.mdl");
		 * model.Export(&file);
		 * @endcode
		*/
		bool Export(std::ofstream*);

		/** @brief Open a file to export with filename, and call bool Model::Export with std::ofstream
		 * @return True if successful, False for incomplete models or corrupt file formats
		 * 
		 * @b Example @b Use: Export file to filename
		 * @code{.cpp}
		 * Markov::Model<char> model;
		 * model.Export("test.mdl");
		 * @endcode
		*/
		bool Export(const char* filename);

		/** @brief Return starter Node
		 * @return starter node with 00 NodeValue
		*/
		Node<NodeStorageType>* StarterNode(){ return starterNode;}

		/** @brief Return a vector of all the edges in the model
		 * @return vector of edges
		*/
		std::vector<Edge<NodeStorageType>*>* Edges(){ return &edges;}
		
		/** @brief Return starter Node
		 * @return starter node with 00 NodeValue
		*/
		std::map<NodeStorageType, Node<NodeStorageType>*>* Nodes(){ return &nodes;}

	private:
		/** @brief Map LeftNode is the Nodes NodeValue
		 * Map RightNode is the node pointer
		*/
		std::map<NodeStorageType, Node<NodeStorageType>*> nodes;

		/** @brief Starter Node of this model. 
		 * 
		*/
		Node<NodeStorageType>* starterNode;

		
		/** @brief A list of all edges in this model. 
		 * 
		*/
		std::vector<Edge<NodeStorageType>*> edges;
	};

};

template <typename NodeStorageType>
Markov::Model<NodeStorageType>::Model() {
	this->starterNode = new Markov::Node<NodeStorageType>(0);
	this->nodes.insert({ 0, this->starterNode });
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(std::ifstream* f) {
	std::string cell;

	char src;
	char target;
	long int oc;

	while (std::getline(*f, cell)) {
		//std::cout << "cell: " << cell << std::endl;
		src = cell[0];
		target = cell[cell.length() - 1];
		char* j;
		oc = std::strtol(cell.substr(2, cell.length() - 2).c_str(),&j,10);
		//std::cout << oc << "\n";
		Markov::Node<NodeStorageType>* srcN;
		Markov::Node<NodeStorageType>* targetN;
		Markov::Edge<NodeStorageType>* e;
		if (this->nodes.find(src) == this->nodes.end()) {
			srcN = new Markov::Node<NodeStorageType>(src);
			this->nodes.insert(std::pair<char, Markov::Node<NodeStorageType>*>(src, srcN));
			//std::cout << "Creating new node at start.\n";
		}
		else {
			srcN = this->nodes.find(src)->second;
		}

		if (this->nodes.find(target) == this->nodes.end()) {
			targetN = new Markov::Node<NodeStorageType>(target);
			this->nodes.insert(std::pair<char, Markov::Node<NodeStorageType>*>(target, targetN));
			//std::cout << "Creating new node at end.\n";
		}
		else {
			targetN = this->nodes.find(target)->second;
		}
		e = srcN->Link(targetN);
		e->AdjustEdge(oc);
		this->edges.push_back(e);

		//std::cout << int(srcN->NodeValue()) << " --" << e->EdgeWeight() << "--> " << int(targetN->NodeValue()) << "\n";


	}

	for (std::pair<unsigned char, Markov::Node<NodeStorageType>*> const& x : this->nodes) {
		//std::cout << "Total edges in EdgesV: " << x.second->edgesV.size() << "\n"; 
		std::sort (x.second->edgesV.begin(), x.second->edgesV.end(), [](Edge<NodeStorageType> *lhs, Edge<NodeStorageType> *rhs)->bool{
			return lhs->EdgeWeight() > rhs->EdgeWeight();
		});
		//for(int i=0;i<x.second->edgesV.size();i++)
		//	std::cout << x.second->edgesV[i]->EdgeWeight() << ", ";
		//std::cout << "\n";
	}
	//std::cout << "Total number of nodes: " << this->nodes.size() << std::endl;
	//std::cout << "Total number of edges: " << this->edges.size() << std::endl;

	return true;
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Import(const char* filename) {
	std::ifstream importfile;
	importfile.open(filename);
	return this->Import(&importfile);

}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(std::ofstream* f) {
	Markov::Edge<NodeStorageType>* e;
	for (std::vector<int>::size_type i = 0; i != this->edges.size(); i++) {
		e = this->edges[i];
		//std::cout << e->LeftNode()->NodeValue() << "," << e->EdgeWeight() << "," << e->RightNode()->NodeValue() << "\n";
		*f << e->LeftNode()->NodeValue() << "," << e->EdgeWeight() << "," << e->RightNode()->NodeValue() << "\n";
	}

	return true;
}

template <typename NodeStorageType>
bool Markov::Model<NodeStorageType>::Export(const char* filename) {
	std::ofstream exportfile;
	exportfile.open(filename);
	return this->Export(&exportfile);
}

template <typename NodeStorageType>
NodeStorageType* Markov::Model<NodeStorageType>::RandomWalk(Markov::Random::RandomEngine* randomEngine, int minSetting, int maxSetting, NodeStorageType* buffer) {
	Markov::Node<NodeStorageType>* n = this->starterNode;
	int len = 0;
	Markov::Node<NodeStorageType>* temp_node;
	while (true) {
		temp_node = n->RandomNext(randomEngine);
		if (len >= maxSetting) {
			break;
		}
		else if ((temp_node == NULL) && (len < minSetting)) {
			continue;
		}

		else if (temp_node == NULL){
			break;
		}
			
		n = temp_node;

		buffer[len++] = n->NodeValue();
	}

	//null terminate the string
	buffer[len] = 0x00;

	//do something with the generated string
	return buffer; //for now
}

template <typename NodeStorageType>
void Markov::Model<NodeStorageType>::AdjustEdge(const NodeStorageType* payload, long int occurrence) {
	NodeStorageType p = payload[0];
	Markov::Node<NodeStorageType>* curnode = this->starterNode;
	Markov::Edge<NodeStorageType>* e;
	int i = 0;

	if (p == 0) return;
	while (p != 0) {
		e = curnode->FindEdge(p);
		if (e == NULL) return;
		e->AdjustEdge(occurrence);
		curnode = e->RightNode();
		p = payload[++i];
	}

	e = curnode->FindEdge('\xff');
	e->AdjustEdge(occurrence);
	return;
}




