
#include <iostream>
#include "color/term.h"

int main(int argc, char** argv) {
	terminal t;
	std::cout << TERM_SUCC << terminal::color::RED  <<  "Library loaded." << terminal::color::RESET << std::endl;


	return 0;
}