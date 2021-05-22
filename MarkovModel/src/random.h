

#include <random>
#include <iostream>
namespace Markov::Random{

class RandomEngine{
public: 
	virtual inline unsigned long random() = 0;
};

class Marsaglia : public RandomEngine{
public:
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
private:
	inline std::random_device& rd() {
		static std::random_device _rd;
		return _rd;
	}

	inline std::default_random_engine& generator() {
		static std::default_random_engine _generator(rd()());
		return _generator;
	}


	inline std::uniform_int_distribution<long long unsigned>& distribution() {
		static std::uniform_int_distribution<long long unsigned> _distribution(0, 0xffffFFFF);
		return _distribution;
	}

	unsigned long x;
	unsigned long y;
	unsigned long z;
};

};