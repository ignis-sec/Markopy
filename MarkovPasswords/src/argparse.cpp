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
	"Usage:	\n"
	"	markov.exe -if empty_model.mdl -ef model.mdl\n"
	"		import empty_model.mdl and train it with data from stdin. When done, output the model to model.mdl\n"
	"\n"
	"	markov.exe -if empty_model.mdl -n 15000 -of wordlist.txt\n"
	"		import empty_model.mdl and generate 15000 words to wordlist.txt\n"
	
	 << std::endl;
}
//new line for the code covarage