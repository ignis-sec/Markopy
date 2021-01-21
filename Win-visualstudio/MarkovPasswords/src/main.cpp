#pragma once
#include <iostream>
#include "color/term.h"
#include "argparse.h"
#include <string>
#include <sstream>
#include "markovPasswords.h"

std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_int_distribution<long long unsigned> distribution(0, 0xffffFFFF);

/** @brief Launch CLI tool.
*/
int main(int argc, char** argv) {

	terminal t;
	
	ProgramOptions* p  = Argparse::parse(argc, argv);

	if (p==0 || p->bFailure) {
		std::cout << TERM_FAIL << "Arguments Failed to Parse" << std::endl;
		Argparse::help();
	}


	MarkovPasswords markovPass;

	

	std::ifstream inputfile;
	inputfile.open("datasets/pwdb.dat", std::ios_base::binary);

	std::string line;
	
	int i = 0;
	
	srand(time(NULL));
	 
	/* GENERATION
	markovPass.Import("models/2gram.mdl");
	std::cout << "0";
	while (std::getline(inputfile, line, '\n')) {
		int oc;
		unsigned char pass[128];
		//dirty cutoff fix for now 
		if (line.size() > 100) line = line.substr(0, 100);

		sscanf_s(line.c_str(), "%d\x09%s", &oc, pass);
		
		i++;
		if(! (i%100000)) std::cout << "\r" << i << ":  " << oc << ":" << pass << "                             ";
		markovPass.adjust(pass, oc);
	}
	markovPass.Export("models/2gram-built.mdl");
	*/

	markovPass.Import("models/2gram-built.mdl");

	char* res;
	char print[100];
	for (int i = 0; i < 100; i++) {
		res = markovPass.RandomWalk();
		strcpy_s(print,100, (char*)res);
		std::cout << "Generation result: " << res << std::endl;
		delete res;
	}

	return 0;
}