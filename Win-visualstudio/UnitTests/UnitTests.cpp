#include "pch.h"
#include "CppUnitTest.h"
#include "../MarkovModel/src/MarkovModel.h"
#include "../MarkovPasswords/src/argparse.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


/** @brief Namespace for Microsoft Native Unit Testing Classes
*/
namespace Testing {

	/** @brief Testing Namespace for Minimal Viable Product
	*/
	namespace MVP {
		/** @brief Testing Namespace for MVP MarkovModel
		*/
		namespace MarkovModel
		{
			/** @brief Test class for minimal viable Edge
			*/
			TEST_CLASS(Edge)
			{
			public:

				/** @brief test default constructor
				*/
				TEST_METHOD(default_constructor) {
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>;
					Assert::IsNull(e->left());
					Assert::IsNull(e->right());
					delete e;
				}

				/** @brief test linked constructor with two nodes
				*/
				TEST_METHOD(linked_constructor) {
					Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(left, right);
					Assert::IsTrue(left == e->left());
					Assert::IsTrue(right == e->right());
					delete left;
					delete right;
					delete e;
				}

				/** @brief test adjust function
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

				/** @brief test traverse returning right
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

				/** @brief test left/right setter
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

				/** @brief test negative adjustments
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
			};

			/** @brief Test class for minimal viable Node
			*/
			TEST_CLASS(Node)
			{
			public:

				/** @brief test default constructor
				*/
				TEST_METHOD(default_constructor) {
					Markov::Node<unsigned char>* n = new Markov::Node<unsigned char>();
					Assert::AreEqual((unsigned char)0, n->value());
					delete n;
				}

				/** @brief test custom constructor with unsigned char
				*/
				TEST_METHOD(uchar_constructor) {
					Markov::Node<unsigned char>* n = NULL;
					unsigned char test_cases[] = { 'c', 0x00, 0xff, -32 };
					for (unsigned char tcase : test_cases) {
						n = new Markov::Node<unsigned char>(tcase);
						Assert::AreEqual(tcase, n->value());
						delete n;
					}
				}

				/** @brief test link function
				*/
				TEST_METHOD(link_left) {
					Markov::Node<unsigned char>* left = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* right = new Markov::Node<unsigned char>('r');

					Markov::Edge<unsigned char>* e = left->Link(right);
					delete left;
					delete right;
					delete e;
				}

				/** @brief test link function
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

				/** @brief test RandomNext with low values
				*/
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

				/** @brief test RandomNext with 32 bit high values
				*/
				TEST_METHOD(rand_next_u32) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Edge<unsigned char>* e = src->Link(target1);
					e->adjust(1 << 31);
					Markov::Node<unsigned char>* res = src->RandomNext();
					Assert::IsTrue(res == target1);
					delete src;
					delete target1;
					delete e;

				}

				/** @brief random next on a node with no follow-ups
				*/
				TEST_METHOD(rand_next_choice_1) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
					Markov::Edge<unsigned char>* e1 = src->Link(target1);
					Markov::Edge<unsigned char>* e2 = src->Link(target2);
					e1->adjust(1);
					e2->adjust((unsigned long)(1ull << 31));
					Markov::Node<unsigned char>* res = src->RandomNext();
					Assert::IsNotNull(res);
					Assert::IsTrue(res == target2);
					delete src;
					delete target1;
					delete e1;
					delete e2;
				}

				/** @brief random next on a node with no follow-ups
				*/
				TEST_METHOD(rand_next_choice_2) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
					Markov::Edge<unsigned char>* e1 = src->Link(target1);
					Markov::Edge<unsigned char>* e2 = src->Link(target2);
					e2->adjust(1);
					e1->adjust((unsigned long)(1ull << 31));
					Markov::Node<unsigned char>* res = src->RandomNext();
					Assert::IsNotNull(res);
					Assert::IsTrue(res == target1);
					delete src;
					delete target1;
					delete e1;
					delete e2;
				}


				/** @brief test updateEdges
				*/
				TEST_METHOD(update_edges_count) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
					Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>(src, target1);
					e1->adjust(25);
					src->UpdateEdges(e1);
					e2->adjust(30);
					src->UpdateEdges(e2);

					Assert::AreEqual((size_t)2, src->Edges()->size());

					delete src;
					delete target1;
					delete e1;
					delete e2;

				}

				/** @brief test updateEdges
				*/
				TEST_METHOD(update_edges_total) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
					Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>(src, target1);
					e1->adjust(25);
					src->UpdateEdges(e1);
					e2->adjust(30);
					src->UpdateEdges(e2);

					Assert::AreEqual(55ull, src->TotalEdgeWeights());

					delete src;
					delete target1;
					delete e1;
					delete e2;
				}


				/** @brief test FindVertice
				*/
				TEST_METHOD(find_vertice) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
					Markov::Edge<unsigned char>* res = NULL;
					src->Link(target1);
					src->Link(target2);

					
					res = src->findEdge('b');
					Assert::IsNotNull(res);
					Assert::AreEqual((unsigned char)'b', res->traverse()->value());
					res = src->findEdge('c');
					Assert::IsNotNull(res);
					Assert::AreEqual((unsigned char)'c', res->traverse()->value());

					delete src;
					delete target1;
					delete target2;


				}


				/** @brief test FindVertice
				*/
				TEST_METHOD(find_vertice_without_any) {

					auto _invalid_next = [] {
						Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
						Markov::Edge<unsigned char>* res = NULL;

						res = src->findEdge('b');
						Assert::IsNull(res);

						delete src;
					};

					Assert::ExpectException<std::logic_error>(_invalid_next);
				}

				/** @brief test FindVertice
				*/
				TEST_METHOD(find_vertice_nonexistent) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
					Markov::Edge<unsigned char>* res = NULL;
					src->Link(target1);
					src->Link(target2);

					res = src->findEdge('D');
					Assert::IsNull(res);

					delete src;
					delete target1;
					delete target2;

				}
			};

			/** @brief Test class for minimal viable Model
			*/
			TEST_CLASS(Model)
			{
			public:
				/** @brief test model constructor for starter node
				*/
				TEST_METHOD(model_constructor) {
					Markov::Model<unsigned char> m;
					Assert::AreEqual((unsigned char)'\0', m.StarterNode()->value());
				}

				/** @brief test import
				*/
				TEST_METHOD(import_filename) {
					Markov::Model<unsigned char> m;
					Assert::IsTrue(m.Import("../MarkovPasswords/Models/2gram.mdl"));
				}

				/** @brief test export
				*/
				TEST_METHOD(export_filename) {
					Markov::Model<unsigned char> m;
					Assert::IsTrue(m.Export("../MarkovPasswords/Models/testcase.mdl"));
				}

				/** @brief test random walk
				*/
				TEST_METHOD(random_walk) {
					Markov::Model<unsigned char> m;
					Assert::IsTrue(m.Import("../MarkovPasswords/Models/2gram.mdl"));
					Assert::IsNotNull(m.RandomWalk());
				}
			};
		}

		/** @brief Testing namespace for MVP MarkovPasswords
		*/
		namespace MarkovPasswords
		{
			/** @brief Test Class for Argparse class
			*/
			TEST_CLASS(ArgParser)
			{
			public:
				/** @brief test basic generate
				*/
				TEST_METHOD(generate_basic) {
					int argc = 3;
					char *argv[] = {"markov.exe", "generate", "-if", "model.mdl", "-of", "passwords.txt", "-n", "100"};

					ProgramOptions *p = Argparse::parse(argc, argv);
					Assert::IsNotNull(p);

					Assert::AreEqual(p->bImport, true);
					Assert::AreEqual(p->bExport, false);
					Assert::AreEqual(p->importname, "model.mdl");
					Assert::AreEqual(p->outputfilename, "passwords.txt");
					Assert::AreEqual(p->generateN, 100);
				}

				/** @brief test basic generate reordered params
				*/
				TEST_METHOD(generate_basic_reorder) {
					int argc = 3;
					char *argv[] = { "markov.exe", "generate", "-n", "100", "-if", "model.mdl", "-of", "passwords.txt" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNotNull(p);

					Assert::AreEqual(p->bImport, true);
					Assert::AreEqual(p->bExport, false);
					Assert::AreEqual(p->importname, "model.mdl");
					Assert::AreEqual(p->outputfilename, "passwords.txt");
					Assert::AreEqual(p->generateN, 100);
				}

				/** @brief test basic generate param longnames
				*/
				TEST_METHOD(generate_basic_longname) {
					int argc = 3;
					char *argv[] = { "markov.exe", "generate", "-n", "100", "--inputfilename", "model.mdl", "--outputfilename", "passwords.txt" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNotNull(p);

					Assert::AreEqual(p->bImport, true);
					Assert::AreEqual(p->bExport, false);
					Assert::AreEqual(p->importname, "model.mdl");
					Assert::AreEqual(p->outputfilename, "passwords.txt");
					Assert::AreEqual(p->generateN, 100);
				}

				/** @brief test basic generate
				*/
				TEST_METHOD(generate_fail_badmethod) {
					int argc = 3;
					char *argv[] = { "markov.exe", "junk", "-n", "100", "--inputfilename", "model.mdl", "--outputfilename", "passwords.txt" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNull(p);
				}

				/** @brief test basic generate
				*/
				TEST_METHOD(train_basic) {
					int argc = 3;
					char *argv[] = { "markov.exe", "train", "-ef", "model.mdl" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNotNull(p);

					Assert::AreEqual(p->bImport, false);
					Assert::AreEqual(p->bExport, true);
					Assert::AreEqual(p->exportname, "model.mdl");

				}

				/** @brief test basic generate
				*/
				TEST_METHOD(train_basic_longname) {
					int argc = 3;
					char *argv[] = { "markov.exe", "train", "--exportfilename", "model.mdl" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNotNull(p);

					Assert::AreEqual(p->bImport, false);
					Assert::AreEqual(p->bExport, true);
					Assert::AreEqual(p->exportname, "model.mdl");
				}

			};

		}
	}


	/** @brief Testing namespace for MarkovModel
	*/
	namespace MarkovModel {
	
		/** @brief Test class for rest of Edge cases
		*/
		TEST_CLASS(Edge)
		{
		public:
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

			/** @brief test integer overflows
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

		/** @brief Test class for rest of Node cases
		*/
		TEST_CLASS(Node)
		{
		public:

			/** @brief test RandomNext with 64 bit high values
			*/
			TEST_METHOD(rand_next_u64) {

				Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
				Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
				Markov::Edge<unsigned char>* e = src->Link(target1);
				e->adjust((unsigned long)(1ull << 63));
				Markov::Node<unsigned char>* res = src->RandomNext();
				Assert::IsTrue(res == target1);
				delete src;
				delete target1;
				delete e;

			}

			/** @brief test RandomNext with 64 bit high values
			*/
			TEST_METHOD(rand_next_u64_max) {

				Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
				Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
				Markov::Edge<unsigned char>* e = src->Link(target1);
				e->adjust((0xffffFFFF));
				Markov::Node<unsigned char>* res = src->RandomNext();
				Assert::IsTrue(res == target1);
				delete src;
				delete target1;
				delete e;

			}

			/** @brief randomNext when no edges are present
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

		/** @brief Test class for rest of model cases
		*/
		TEST_CLASS(Model)
		{
		public:
			TEST_METHOD(functional_random_walk) {
				Markov::Model<unsigned char> m;
				Markov::Node<unsigned char>* starter = m.StarterNode();
				Markov::Node<unsigned char>* a = new Markov::Node<unsigned char>('a');
				Markov::Node<unsigned char>* b = new Markov::Node<unsigned char>('b');
				Markov::Node<unsigned char>* c = new Markov::Node<unsigned char>('c');
				Markov::Node<unsigned char>* end = new Markov::Node<unsigned char>(0xff);
				starter->Link(a)->adjust(1);
				a->Link(b)->adjust(1);
				b->Link(c)->adjust(1);
				c->Link(end)->adjust(1);

				char* res = (char*)m.RandomWalk();
				Assert::IsFalse(strcmp(res, "abc"));
			}
		};

	}

	/** @brief Testing namespace for MarkovPasswords
	*/
	namespace MarkovPasswords {

	};

}