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
	@brief Namespace for model related classes.
*/
namespace Markov {
	/** @brief class for the final Markov Model, constructed from nodes and edges. 
	* 
	* This class will be *templated later to work with other data types than char*.
	*/
	template <typename NodeStorageType>
	class Node;

	template <typename NodeStorageType>
	class Edge;

	template <typename NodeStorageType>
	class Model {
	public:
		
		/** @brief Initialize a model with only start and end nodes.
		*/
		Model<NodeStorageType>();
			
		/** @brief Do a random walk on this model. 
		* Start from the starter node, invoke RandomNext on current node until terminator node is reached.
		* @return Null terminated string that was generated.
		*/
		NodeStorageType* RandomWalk(Markov::Random::RandomEngine* randomEngine, int minSetting, int maxSetting, NodeStorageType* buffer);

		/** @brief Adjust the model with a single string. 
		* Start from the starter node, and for each character, AdjustEdge the edge EdgeWeight from current node to the next, until NULL character is reached.
		* Then, update the edge EdgeWeight from current node, to the terminator node.
		* @param string - String that is passed from the training, and will be used to AdjustEdge the model with
		* @param occurrence - Occurrence of this string. 
		*/
		void AdjustEdge(const NodeStorageType* payload, long int occurrence);

		/** @brief Import a file to construct the model. 
		* 
		* File contains a list of edges.
		* Format is: Left_repr;EdgeWeight;right_repr
		* Iterate over this list, and construct nodes and edges accordingly. 
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Import(std::ifstream*);

		/** @brief Open a file to import with filename, and call bool Model::Import with std::ifstream
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Import(const char* filename);

		/** @brief Export a file of the model.
		*
		* File contains a list of edges.
		* Format is: Left_repr;EdgeWeight;right_repr
		* Iterate over this vertices, and their edges, and write them to file.
		* @return True if successful, False for incomplete models.
		*/
		bool Export(std::ofstream*);

		/** @brief Open a file to export with filename, and call bool Model::Export with std::ofstream
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Export(const char* filename);

		/** @brief Return starter Node
		* @return starter node with 00 NodeValue
		*/
		Node<NodeStorageType>* StarterNode(){ return starterNode;}

		std::vector<Edge<NodeStorageType>*>* Edges(){ return &edges;}

		std::map<NodeStorageType, Node<NodeStorageType>*>* Nodes(){ return &nodes;}

	private:
		/** @brief Map LeftNode is the Nodes NodeValue
		* Map RightNode is the node pointer
		*/
		std::map<NodeStorageType, Node<NodeStorageType>*> nodes;

		/** @brief Starter Node of this model. */
		Node<NodeStorageType>* starterNode;

		
		/** @brief A list of all edges in this model. */
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




