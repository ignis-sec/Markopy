#include "pch.h"
#include "CppUnitTest.h"
#include "../MarkovModel/src/MarkovModel.h"
#include "../MarkovPasswords/src/argparse.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_int_distribution<long long unsigned> distribution(0, 0xffffFFFF);

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
					Assert::IsNull(e->LeftNode());
					Assert::IsNull(e->RightNode());
					delete e;
				}

				/** @brief test linked constructor with two nodes
				*/
				TEST_METHOD(linked_constructor) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					Assert::IsTrue(LeftNode == e->LeftNode());
					Assert::IsTrue(RightNode == e->RightNode());
					delete LeftNode;
					delete RightNode;
					delete e;
				}

				/** @brief test AdjustEdge function
				*/
				TEST_METHOD(AdjustEdge) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					e->AdjustEdge(15);
					Assert::AreEqual(15ull, e->EdgeWeight());
					e->AdjustEdge(15);
					Assert::AreEqual(30ull, e->EdgeWeight());
					delete LeftNode;
					delete RightNode;
					delete e;
				}

				/** @brief test TraverseNode returning RightNode
				*/
				TEST_METHOD(TraverseNode) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					Assert::IsTrue(RightNode == e->TraverseNode());
					delete LeftNode;
					delete RightNode;
					delete e;
				}

				/** @brief test LeftNode/RightNode setter
				*/
				TEST_METHOD(set_left_and_right) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(LeftNode, RightNode);

					Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>;
					e2->SetLeftEdge(LeftNode);
					e2->SetRightEdge(RightNode);

					Assert::IsTrue(e1->LeftNode() == e2->LeftNode());
					Assert::IsTrue(e1->RightNode() == e2->RightNode());
					delete LeftNode;
					delete RightNode;
					delete e1;
					delete e2;
				}

				/** @brief test negative adjustments
				*/
				TEST_METHOD(negative_adjust) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					e->AdjustEdge(15);
					Assert::AreEqual(15ull, e->EdgeWeight());
					e->AdjustEdge(-15);
					Assert::AreEqual(0ull, e->EdgeWeight());
					delete LeftNode;
					delete RightNode;
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
					Assert::AreEqual((unsigned char)0, n->NodeValue());
					delete n;
				}

				/** @brief test custom constructor with unsigned char
				*/
				TEST_METHOD(uchar_constructor) {
					Markov::Node<unsigned char>* n = NULL;
					unsigned char test_cases[] = { 'c', 0x00, 0xff, -32 };
					for (unsigned char tcase : test_cases) {
						n = new Markov::Node<unsigned char>(tcase);
						Assert::AreEqual(tcase, n->NodeValue());
						delete n;
					}
				}

				/** @brief test link function
				*/
				TEST_METHOD(link_left) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');

					Markov::Edge<unsigned char>* e = LeftNode->Link(RightNode);
					delete LeftNode;
					delete RightNode;
					delete e;
				}

				/** @brief test link function
				*/
				TEST_METHOD(link_right) {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');

					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(NULL, RightNode);
					LeftNode->Link(e);
					Assert::IsTrue(LeftNode == e->LeftNode());
					Assert::IsTrue(RightNode == e->RightNode());
					delete LeftNode;
					delete RightNode;
					delete e;
				}

				/** @brief test RandomNext with low values
				*/
				TEST_METHOD(rand_next_low) {

					Markov::Node<unsigned char>* src = new Markov::Node<unsigned char>('a');
					Markov::Node<unsigned char>* target1 = new Markov::Node<unsigned char>('b');
					Markov::Edge<unsigned char>* e = src->Link(target1);
					e->AdjustEdge(15);
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
					e->AdjustEdge(1 << 31);
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
					e1->AdjustEdge(1);
					e2->AdjustEdge((unsigned long)(1ull << 31));
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
					e2->AdjustEdge(1);
					e1->AdjustEdge((unsigned long)(1ull << 31));
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
					Markov::Node<unsigned char>* target2 = new Markov::Node<unsigned char>('c');
					Markov::Edge<unsigned char>* e1 = new Markov::Edge<unsigned char>(src, target1);
					Markov::Edge<unsigned char>* e2 = new Markov::Edge<unsigned char>(src, target2);
					e1->AdjustEdge(25);
					src->UpdateEdges(e1);
					e2->AdjustEdge(30);
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
					e1->AdjustEdge(25);
					src->UpdateEdges(e1);
					e2->AdjustEdge(30);
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

					
					res = src->FindEdge('b');
					Assert::IsNotNull(res);
					Assert::AreEqual((unsigned char)'b', res->TraverseNode()->NodeValue());
					res = src->FindEdge('c');
					Assert::IsNotNull(res);
					Assert::AreEqual((unsigned char)'c', res->TraverseNode()->NodeValue());

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

						res = src->FindEdge('b');
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

					res = src->FindEdge('D');
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
					Assert::AreEqual((unsigned char)'\0', m.StarterNode()->NodeValue());
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
					int argc = 8;
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
					int argc = 8;
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
					int argc = 8;
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
					int argc = 8;
					char *argv[] = { "markov.exe", "junk", "-n", "100", "--inputfilename", "model.mdl", "--outputfilename", "passwords.txt" };

					ProgramOptions* p = Argparse::parse(argc, argv);
					Assert::IsNull(p);
				}

				/** @brief test basic generate
				*/
				TEST_METHOD(train_basic) {
					int argc = 4;
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
					int argc = 4;
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
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					e->AdjustEdge(15);
					e->AdjustEdge(-30);
					delete LeftNode;
					delete RightNode;
					delete e;
				};
				Assert::ExpectException<std::underflow_error>(_underflow_adjust);
			}

			/** @brief test integer overflows
			*/
			TEST_METHOD(except_integer_overflow) {
				auto _overflow_adjust = [] {
					Markov::Node<unsigned char>* LeftNode = new Markov::Node<unsigned char>('l');
					Markov::Node<unsigned char>* RightNode = new Markov::Node<unsigned char>('r');
					Markov::Edge<unsigned char>* e = new Markov::Edge<unsigned char>(LeftNode, RightNode);
					e->AdjustEdge(~0ull);
					e->AdjustEdge(1);
					delete LeftNode;
					delete RightNode;
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
				e->AdjustEdge((unsigned long)(1ull << 63));
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
				e->AdjustEdge((0xffffFFFF));
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
				starter->Link(a)->AdjustEdge(1);
				a->Link(b)->AdjustEdge(1);
				b->Link(c)->AdjustEdge(1);
				c->Link(end)->AdjustEdge(1);

				char* res = (char*)m.RandomWalk();
				Assert::IsFalse(strcmp(res, "abc"));
			}
			TEST_METHOD(functionoal_random_walk_without_any) {
				Markov::Model<unsigned char> m;
				Markov::Node<unsigned char>* starter = m.StarterNode();
				Markov::Node<unsigned char>* a = new Markov::Node<unsigned char>('a');
				Markov::Node<unsigned char>* b = new Markov::Node<unsigned char>('b');
				Markov::Node<unsigned char>* c = new Markov::Node<unsigned char>('c');
				Markov::Node<unsigned char>* end = new Markov::Node<unsigned char>(0xff);
				Markov::Edge<unsigned char>* res = NULL;
				starter->Link(a)->AdjustEdge(1);
				a->Link(b)->AdjustEdge(1);
				b->Link(c)->AdjustEdge(1);
				c->Link(end)->AdjustEdge(1);

				res = starter->FindEdge('D');
				Assert::IsNull(res);

			}
		};

	}

	/** @brief Testing namespace for MarkovPasswords
	*/
	namespace MarkovPasswords {

	};

}

