#include "pch.h"
#include "CppUnitTest.h"
#include "../MarkovModel/src/MarkovModel.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace MarkovModel
{


	TEST_CLASS(Edge)
	{
	public:

		/** @brief check default constructor
		*/
		TEST_METHOD(default_constructor) {
			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>;
			Assert::IsNull(e->left());
			Assert::IsNull(e->right());
			delete e;
		}

		/** @brief check linked constructor with two nodes
		*/
		TEST_METHOD(linked_constructor) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
			Assert::IsTrue(left  == e->left());
			Assert::IsTrue(right == e->right());
			delete left;
			delete right;
			delete e;
		}

		/** @brief check adjust function
		*/
		TEST_METHOD(adjust) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
			e->adjust(15);
			Assert::AreEqual(15ull, e->weight());
			e->adjust(15);
			Assert::AreEqual(30ull, e->weight());
			delete left;
			delete right;
			delete e;
		}

		/** @brief check traverse returning right
		*/
		TEST_METHOD(traverse) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
			Assert::IsTrue(right == e->traverse());
			delete left;
			delete right;
			delete e;
		}

		/** @brief check left/right setter
		*/
		TEST_METHOD(set_left_and_right) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
			Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(left, right);

			Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>;
			e2->set_left(left);
			e2->set_right(right);

			Assert::IsTrue(e1->left() == e2->left());
			Assert::IsTrue(e1->right() == e2->right());
			delete left;
			delete right;
			delete e1;
			delete e2;
		}

		/** @brief check negative adjustments
		*/
		TEST_METHOD(negative_adjust) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
			e->adjust(15);
			Assert::AreEqual(15ull, e->weight());
			e->adjust(-15);
			Assert::AreEqual(0ull, e->weight());
			delete left;
			delete right;
			delete e;
		}

		/** @brief send exception on integer underflow
		*/
		TEST_METHOD(except_integer_underflow) {
			auto _underflow_adjust = [] {
				Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
				Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
				Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
				e->adjust(15);
				e->adjust(-30);
				delete left;
				delete right;
				delete e;
			};
			Assert::ExpectException<std::underflow_error>(_underflow_adjust);
		}

		/** @brief check integer overflows
		*/
		TEST_METHOD(except_integer_overflow) {
			auto _overflow_adjust = [] {
				Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
				Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
				Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
				e->adjust(~0ull);
				e->adjust(1);
				delete left;
				delete right;
				delete e;
			};
			Assert::ExpectException<std::underflow_error>(_overflow_adjust);
		}
	};

	TEST_CLASS(Node)
	{
	public:
		
		/** @brief check default constructor
		*/
		TEST_METHOD(default_constructor) {
			Markov::Node<unsigned char>* n = new Markov::Node<unsigned char>();
			Assert::AreEqual((unsigned char)0, n->value());
			delete n;
		}

		/** @brief check custom constructor with unsigned char
		*/
		TEST_METHOD(uchar_constructor){
			Markov::Node<unsigned char> *n = NULL;
			unsigned char test_cases[] = { 'c', 0x00, 0xff, -32 };
			for (unsigned char tcase : test_cases) {
				n = new Markov::Node<unsigned char>(tcase);
				Assert::AreEqual(tcase, n->value());
				delete n;
			}
		}

		/** @brief Check link function
		*/
		TEST_METHOD(link_left) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');

			Markov::Edge<unsigned char>* e = left->Link(right);
			delete left;
			delete right;
			delete e;
		}

		/** @brief Check link function
		*/
		TEST_METHOD(link_right) {
			Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
			Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');

			Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(NULL, right);
			left->Link(e);
			Assert::IsTrue(left == e->left());
			Assert::IsTrue(right == e->right());
			delete left;
			delete right;
			delete e;
		}

		TEST_METHOD(rand_next_low) {

			Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
			Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
			Markov::Edge<unsigned char>* e = src->Link(target1);
			e->adjust(15);
			Markov::Node<unsigned char>* res = src->RandomNext();
			Assert::IsTrue(res == target1);
			delete src;
			delete target1;
			delete e;

		}
		 
		TEST_METHOD(rand_next_high) {

			Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
			Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
			Markov::Edge<unsigned char>* e = src->Link(target1);
			e->adjust((unsigned long)(1<<64)-1ull);
			Markov::Node<unsigned char>* res = src->RandomNext();
			Assert::IsTrue(res == target1);
			delete src;
			delete target1;
			delete e;

		}

		/** @brief random next on a node with no follow-ups
		*/
		TEST_METHOD(uninitialized_rand_next) {

			auto _invalid_next = [] {
				Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
				Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
				Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(src, target1);
				Markov::Node<unsigned char>* res = src->RandomNext();

				delete src;
				delete target1;
				delete e;
			};
			
			Assert::ExpectException<std::logic_error>(_invalid_next);
		}
	};
}


namespace MarkovPasswords
{


}