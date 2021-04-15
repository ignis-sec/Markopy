#include "argparse.h"
#include "color/term.h"

ProgramOptions* Argparse::parse(int argc, char** argv) { return 0; }

void Argparse::help() {
	std::cout << 
	"Markov Passwords - Help\n"
	"Options:\n"
	"	\n"
	"	-of --outputfilename\n" 
	" 		Filename to output the generation results\n"
	"	-ef --exportfilename\n"
	" 		filename to export built model to\n"
	"	-if --importfilename\n"
	" 		filename to import model from\n"
	"	-n (generate count)\n"
	" 		Number of lines to generate\n"
	"	\n"
	
	 << std::endl;
}