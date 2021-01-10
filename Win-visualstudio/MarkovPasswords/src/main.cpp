
#include <iostream>
#include "color/term.h"
#include "argparse.h"
#include "markovPasswords.h"
#include <fstream>


/** @brief Launch CLI tool.
*/
int main(int argc, char** argv) {

	terminal t;

	
	ProgramOptions* p  = Argparse::parse(argc, argv);

	if (p==0 || p->bFailure) {
		std::cout << TERM_FAIL << "Arguments Failed to Parse" << std::endl;
		Argparse::help();
	}


	return 0;
}