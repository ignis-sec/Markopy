#pragma once
#include <iostream>
#include "color/term.h"
#include "argparse.h"
#include <string>
#include <cstring>
#include <sstream>
#include "markovPasswords.h"

std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_int_distribution<long long unsigned> distribution(0, 0xffffFFFF);

/** @brief Launch CLI tool.
*/
int main(int argc, char** argv) {

	terminal t;
	/*
	ProgramOptions* p  = Argparse::parse(argc, argv);

	if (p==0 || p->bFailure) {
		std::cout << TERM_FAIL << "Arguments Failed to Parse" << std::endl;
		Argparse::help();
	}*/


	MarkovPasswords markovPass;
	std::cout << "Importing model.\n";
	//markovPass.Import("models/2gram.mdl");
	std::cout << "Import done. Training...\n";
	//markovPass.Train("datasets/dataset.5.dat", '\t');
	std::cout << "Training done. Exporting to file.\n";
	//markovPass.Export("models/finished.mdl");

	std::cout << "Exported. Generating....\n";
	markovPass.Import("models/finished.mdl");
	markovPass.Generate(500, "datasets/output.txt");

	std::cout << "Generation done....\n";
	return 0;
}