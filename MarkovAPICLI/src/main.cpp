/** @file main.cpp
 * @brief Test cases for Markov::API::ModelMatrix
 * @authors Ata Hakçıl, Celal Sahir Çetiner
 * 
 * @copydoc Markov::API::CLI::Argparse
 * @copydoc Markov::API::ModelMatrix
 * @copydoc Markov::API::MarkovPasswords
 */

#pragma once
#include <iostream>
#include "color/term.h"
#include "argparse.h"
#include <string>
#include <cstring>
#include <sstream>
#include "MarkovAPI/src/markovPasswords.h"
#include "MarkovAPI/src/modelMatrix.h"
#include <chrono>

/** @brief Launch CLI tool.
*/
int main(int argc, char** argv) {

	Markov::API::CLI::Terminal t;
	/*
	ProgramOptions* p  = Argparse::parse(argc, argv);

	if (p==0 || p->bFailure) {
		std::cout << TERM_FAIL << "Arguments Failed to Parse" << std::endl;
		Argparse::help();
	}*/
	Markov::API::CLI::Argparse a(argc,argv);

	Markov::API::ModelMatrix markovPass;
	std::cerr << "Importing model.\n";
	markovPass.Import("models/finished.mdl");
	std::cerr << "Import done. \n";

	markovPass.Buff("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 100);
	
	markovPass.Export("models/buffed-symbols-100fold.mdl");
	//markovPass.ConstructMatrix();
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//markovPass.FastRandomWalk(50000000,"/media/ignis/Stuff/wordlist.txt",6,12,25, true);
	//markovPass.FastRandomWalk(500000000,"/media/ignis/Stuff/wordlist2.txt",6,12,25, true);
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	//std::cerr << "Finished in:" << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " milliseconds" << std::endl;
	return 0;
}

