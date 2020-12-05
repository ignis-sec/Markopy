
#include <iostream>
#include "color/term.h"


void dll_loadtest();   // a function from a DLL


int main(int argc, char** argv) {
	terminal t;
	std::cout << TERM_SUCC << terminal::color::RED  <<  "Library loaded." << terminal::color::RESET << std::endl;
	dll_loadtest();

	return 0;
}