
/** @file random.h
 * @brief Random engine implementations for Markov
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::Random::RandomEngine
 * @copydoc Markov::Random::DefaultRandomEngine
 * @copydoc Markov::Random::Marsaglia
 */

#pragma once
#include <random>
#include <iostream>

/**
	@brief Objects related to RNG
*/
namespace Markov::Random{

	/** @brief An abstract class for Random Engine
	 * 
	 * This class is used for generating random numbers, which are used for random walking on the graph.
	 * 
	 * Main reason behind allowing different random engines is that some use cases may favor performance,
	 * while some favor good random. 
	 * 
	 * Mersenne can be used for truer random, while Marsaglia can be used for deterministic but fast random.
	 * 
	*/
	class RandomEngine{
	public: 
		virtual inline unsigned long random() = 0;
	};



	/** @brief Implementation using Random.h default random engine
	 * 
	 * This engine is also used by other engines for seeding.
	 * 
	 * 
	 * @b Example @b Use: Using Default Engine with RandomWalk
	 * @code{.cpp}
	 * Markov::Model<char> model;
	 * Model.import("model.mdl");
	 * char* res = new char[11];
	 * Markov::Random::DefaultRandomEngine randomEngine;
	 * for (int i = 0; i < 10; i++) {
	 *		this->RandomWalk(&randomEngine, 5, 10, res); 
	 *		std::cout << res << "\n";
	 *	}
	 * @endcode
	 * 
	 * @b Example @b Use: Generating a random number with Marsaglia Engine
	 * @code{.cpp}
	 * Markov::Random::DefaultRandomEngine de;
	 * std::cout << de.random();
	 * @endcode
	 * 
	*/
	class DefaultRandomEngine : public RandomEngine{
	public:
		/** @brief Generate Random Number
		* @return random number in long range.
		*/
		inline unsigned long random(){
			return this->distribution()(this->generator());
		}
	protected:

		/** @brief Default random device for seeding
		* 
		*/
		inline std::random_device& rd() {
			static std::random_device _rd;
			return _rd;
		}
		
		/** @brief Default random engine for seeding
		* 
		*/
		inline std::default_random_engine& generator() {
			static std::default_random_engine _generator(rd()());
			return _generator;
		}

		/** @brief Distribution schema for seeding.
		* 
		*/
		inline std::uniform_int_distribution<long long unsigned>& distribution() {
			static std::uniform_int_distribution<long long unsigned> _distribution(0, 0xffffFFFF);
			return _distribution;
		}

	};


	/** @brief Implementation of Marsaglia Random Engine
	 * 
	 * This is an implementation of Marsaglia Random engine, which for most use cases is a better fit than other solutions.
	 * Very simple mathematical formula to generate pseudorandom integer, so its crazy fast.
	 * 
	 * This implementation of the Marsaglia Engine is seeded by random.h default random engine.
	 * RandomEngine is only seeded once so its not a performance issue.
	 * 
	 * @b Example @b Use: Using Marsaglia Engine with RandomWalk
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
	 * @b Example @b Use: Generating a random number with Marsaglia Engine
	 * @code{.cpp}
	 * Markov::Random::Marsaglia me;
	 * std::cout << me.random();
	 * @endcode
	 * 
	*/
	class Marsaglia : public DefaultRandomEngine{
	public:

		/** @brief Construct Marsaglia Engine
		* 
		* Initialize x,y and z using the default random engine.
		*/
		Marsaglia(){
			this->x = this->distribution()(this->generator());
			this->y = this->distribution()(this->generator());
			this->z = this->distribution()(this->generator());
			//std::cout << "x: " << x << ", y: " << y << ", z: " << z << "\n";
		}


	inline unsigned long random(){	
		unsigned long t;
		x ^= x << 16;
		x ^= x >> 5;
		x ^= x << 1;

		t = x;
		x = y;
		y = z;
		z = t ^ x ^ y;

		return z;
	}
	

		unsigned long x;
		unsigned long y;
		unsigned long z;
	};


	/** @brief Implementation of Mersenne Twister Engine
	 * 
	 * This is an implementation of Mersenne Twister Engine, which is slow but is a good implementation for high entropy pseudorandom.
	 * 
	 * 
	 * @b Example @b Use: Using Mersenne Engine with RandomWalk
	 * @code{.cpp}
	 * Markov::Model<char> model;
	 * Model.import("model.mdl");
	 * char* res = new char[11];
	 * Markov::Random::Mersenne MersenneTwisterEngine;
	 * for (int i = 0; i < 10; i++) {
	 *		this->RandomWalk(&MersenneTwisterEngine, 5, 10, res); 
	 *		std::cout << res << "\n";
	 *	}
	 * @endcode
	 * 
	 * @b Example @b Use: Generating a random number with Marsaglia Engine
	 * @code{.cpp}
	 * Markov::Random::Mersenne me;
	 * std::cout << me.random();
	 * @endcode
	 * 
	*/
	class Mersenne : public DefaultRandomEngine{

	};


};