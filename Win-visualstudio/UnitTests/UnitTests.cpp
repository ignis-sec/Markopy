#include "pch.h"
#include "CppUnitTest.h"
#include "../MarkovModel/src/MarkovModel.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace UnitTests
{
	TEST_CLASS(UnitTests)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			unsigned char name = 'c';
			//Markov::Node<unsigned char> n(name);
			Markov::Model<unsigned char> m;
		}
	};
}
