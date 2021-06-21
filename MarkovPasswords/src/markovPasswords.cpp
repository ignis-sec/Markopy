#pragma once
#include "markovPasswords.h"
#include <string.h>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <string>
#include <signal.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

static volatile int keepRunning = 1;

void intHandler(int dummy) {
	std::cout << "You wanted this man by presing CTRL-C ! Ok bye.";
	//Sleep(5000);
	keepRunning = 0;
	exit(0);
}


Markov::API::MarkovPasswords::MarkovPasswords() : Markov::Model<char>(){
	
	
}

Markov::API::MarkovPasswords::MarkovPasswords(const char* filename) {
	
	std::ifstream* importFile;

    this->Import(filename);
	
	//std::ifstream* newFile(filename);
	
	//importFile = newFile;	

}

std::ifstream* Markov::API::MarkovPasswords::OpenDatasetFile(const char* filename){

	std::ifstream* datasetFile;

	std::ifstream newFile(filename);

	datasetFile = &newFile;

	this->Import(datasetFile);
	return datasetFile;
}



void Markov::API::MarkovPasswords::Train(const char* datasetFileName, char delimiter, int threads)   {
  signal(SIGINT, intHandler);
	Markov::API::Concurrency::ThreadSharedListHandler listhandler(datasetFileName);
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::thread*> threadsV;
	for(int i=0;i<threads;i++){
		threadsV.push_back(new std::thread(&Markov::API::MarkovPasswords::TrainThread, this, &listhandler, delimiter));
	}

	for(int i=0;i<threads;i++){
		threadsV[i]->join();
		delete threadsV[i];
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

}

void Markov::API::MarkovPasswords::TrainThread(Markov::API::Concurrency::ThreadSharedListHandler *listhandler, char delimiter){
	char format_str[] ="%ld,%s";
	format_str[3]=delimiter;
	std::string line;
	while (listhandler->next(&line) && keepRunning) {
		long int oc;
		if (line.size() > 100) {
			line = line.substr(0, 100);
		}
		char* linebuf = new char[line.length()+5];
#ifdef _WIN32
		sscanf_s(line.c_str(), "%ld,%s", &oc, linebuf, line.length()+5); //<== changed format_str to-> "%ld,%s"
#else
		sscanf(line.c_str(), format_str, &oc, linebuf);
#endif
		this->AdjustEdge((const char*)linebuf, oc); 
		delete linebuf;
	}
}


std::ofstream* Markov::API::MarkovPasswords::Save(const char* filename) {
	std::ofstream* exportFile;

	std::ofstream newFile(filename);

	exportFile = &newFile;
	
	this->Export(exportFile);
	return exportFile;
}


void Markov::API::MarkovPasswords::Generate(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads)  {
	char* res;
	char print[100];
	std::ofstream wordlist;	
	wordlist.open(wordlistFileName);
	std::mutex mlock;
	int iterationsPerThread = n/threads;
	int iterationsCarryOver = n%threads;
	std::vector<std::thread*> threadsV;
	for(int i=0;i<threads;i++){
		threadsV.push_back(new std::thread(&Markov::API::MarkovPasswords::GenerateThread, this, &mlock, iterationsPerThread, &wordlist, minLen, maxLen));
	}

	for(int i=0;i<threads;i++){
		threadsV[i]->join();
		delete threadsV[i];
	}

	this->GenerateThread(&mlock, iterationsCarryOver, &wordlist, minLen, maxLen);
	
}

void Markov::API::MarkovPasswords::GenerateThread(std::mutex *outputLock, unsigned long int n, std::ofstream *wordlist, int minLen, int maxLen)  {
	char* res = new char[maxLen+5];
	if(n==0) return;

	Markov::Random::Marsaglia MarsagliaRandomEngine;
	for (int i = 0; i < n; i++) {
		this->RandomWalk(&MarsagliaRandomEngine, minLen, maxLen, res); 
		outputLock->lock();
		*wordlist << res << "\n";
		outputLock->unlock();
	}
}

void Markov::API::MarkovPasswords::Buff(const char* str, double multiplier, bool bDontAdjustSelfLoops, bool bDontAdjustExtendedLoops){
	std::string buffstr(str);
	std::map< char, Node< char > * > *nodes;
	std::map< char, Edge< char > * > *edges;
    nodes = this->Nodes();
    int i=0;
    for (auto const& [repr, node] : *nodes){
		edges = node->Edges();
		for (auto const& [targetrepr, edge] : *edges){
			if(buffstr.find(targetrepr)!= std::string::npos){
				if(bDontAdjustSelfLoops && repr==targetrepr) continue;
				if(bDontAdjustExtendedLoops){
					if(buffstr.find(repr)!= std::string::npos){
						continue;
					}
				}
				long int weight = edge->EdgeWeight();
				weight = weight*multiplier;		
				edge->AdjustEdge(weight);
			}

        }
        i++;
    }

	this->OptimizeEdgeOrder();
}
