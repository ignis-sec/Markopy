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
	markovPass.Import("models/2gram.mdl");
	markovPass.Train("models/finished.mdl", "datasets/dataset.5.dat", '\t');

	//markovPass.Import("models/finished.mdl");
	markovPass.Generate(500, "dataset/output.txt");
	return 0;
}