/** @dir Model.h
 *
 */

#pragma once
#include <map>
#include <vector>
#include <fstream>

#include "edge.h"
#include "node.h"

/**
	@brief Namespace for model related classes.
*/
namespace Markov {
	/** @brief class for the final Markov Model, constructed from nodes and edges. 
	* 
	* This class will be *templated later to work with other data types than char*.
	*/
	template <typename NodeStorageType>
	class Model {
	public:
		
		/** @brief Initialize a model with only start and end nodes.
		*/
		Markov::Model<NodeStorageType>();

		/** @brief Do a random walk on this model. 
		* Start from the starter node, invoke RandomNext on current node until terminator node is reached.
		* @return Null terminated string that was generated.
		*/
		NodeStorageType* RandomWalk();

		/** @brief Adjust the model with a single string. 
		* Start from the starter node, and for each character, adjust the edge weight from current node to the next, until NULL character is reached.
		* Then, update the edge weight from current node, to the terminator node.
		* @param string - String that is passed from the training, and will be used to adjust the model with
		* @param occurrence - Occurrence of this string. 
		*/
		void adjust(NodeStorageType* payload, long int occurrence);

		/** @brief Import a file to construct the model. 
		* 
		* File contains a list of edges.
		* Format is: Left_repr;weight;right_repr
		* Iterate over this list, and construct nodes and edges accordingly. 
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Import(std::ifstream*);

		/** @brief Open a file to import with filename, and call bool Markov::Model::Import with std::ifstream
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Import(char* filename);

		/** @brief Export a file of the model.
		*
		* File contains a list of edges.
		* Format is: Left_repr;weight;right_repr
		* Iterate over this vertices, and their edges, and write them to file.
		* @return True if successful, False for incomplete models.
		*/
		bool Export(std::ofstream*);

		/** @brief Open a file to export with filename, and call bool Markov::Model::Export with std::ofstream
		* @return True if successful, False for incomplete models or corrupt file formats
		*/
		bool Export(char* filename);

		/** @brief Return starter Node
		* @return starter node with 00 value
		*/
		Markov::Node<NodeStorageType>* StarterNode(){ return starterNode;}

		std::vector<Markov::Edge<NodeStorageType>>* Edges(){ return &edges;}

		std::map<NodeStorageType, Markov::Node<NodeStorageType>*>* Nodes(){ return &nodes;}

	private:
		/** @brief Map left is the Nodes value
		* Map right is the node pointer
		*/
		std::map<NodeStorageType, Markov::Node<NodeStorageType>*> nodes;

		/** @brief Starter Node of this model. */
		Markov::Node<NodeStorageType>* starterNode;

		
		/** @brief A list of all edges in this model. */
		std::vector<Markov::Edge<NodeStorageType>> edges;
	};

};