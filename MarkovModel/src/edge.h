#include <cstdint>


namespace Markov {

	template <typename NodeStorageType>
	class Node;
	/** @brief Edge class used to link nodes in the model together.
	* 
	Has left, right, and weight of the edge.
	Edges are *UNIDIRECTIONAL* in this model. They can only be traversed left to right.
	*/
	template <typename NodeStorageType>
	class Edge {
	public:
		
		/** @brief Default constructor. 
 		*/
		Edge<NodeStorageType>();
		/**@brief Constructor. Initialize edge with given right and left
		* @param _left - Left node of this edge.
		* @param _right - Right node of this edge.
		*/
		
		Edge<NodeStorageType>(Node<NodeStorageType>* _left, Node<NodeStorageType>* _right);
		
		/** @brief Adjust the edge weight with offset.
		* Adds the offset parameter to the edge weight.
		* @param offset - value to be added to the weight
		*/
		void adjust(uint64_t offset);
		
		/** @brief Traverse this edge to right.
		* @return Right node. If this is a terminator node, return NULL
		*/
		Node<NodeStorageType>* traverse();

		/** @brief Set left of this edge.
		* @param node - Node to be linked with.
		*/
		void set_left (Node<NodeStorageType>*);
		/** @brief Set right of this edge.
		* @param node - Node to be linked with.
		*/
		void set_right(Node<NodeStorageType>*);
		
		/** @brief return edge's weight.
		* @return edge's weight.
		*/
		uint64_t weight();

		/** @brief return edge's left
		* @return edge's left.
		*/
		Node<NodeStorageType>* left();

		/** @brief return edge's right
		* @return edge's right.
		*/
		Node<NodeStorageType>* right();

	private:
		Node<NodeStorageType>* _left; /** @brief source node*/
		Node<NodeStorageType>* _right;/** @brief target node*/
		int _weight;    /** @brief Edge weight*/
	};


};