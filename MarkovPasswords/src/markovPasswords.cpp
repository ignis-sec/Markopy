#pragma once
#include "markovPasswords.h"
#include <string.h>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <string>

MarkovPasswords::MarkovPasswords() : Markov::Model<char>(){
	
	
}

MarkovPasswords::MarkovPasswords(const char* filename) {
	
	std::ifstream* importFile;

    this->Import(filename);
	
	//std::ifstream* newFile(filename);
	
	//importFile = newFile;	

}

std::ifstream* MarkovPasswords::OpenDatasetFile(const char* filename){

	std::ifstream* datasetFile;

	std::ifstream newFile(filename);

	datasetFile = &newFile;

	this->Import(datasetFile);
	return datasetFile;
}


void MarkovPasswords::Train(const char* datasetFileName, char delimiter, int threads)   {
	ThreadSharedListHandler listhandler(datasetFileName);
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::thread*> threadsV;
	for(int i=0;i<threads;i++){
		threadsV.push_back(new std::thread(&MarkovPasswords::TrainThread, this, &listhandler, datasetFileName, delimiter));
	}

	for(int i=0;i<threads;i++){
		threadsV[i]->join();
		delete threadsV[i];
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	
}

void MarkovPasswords::TrainThread(ThreadSharedListHandler *listhandler, const char* datasetFileName, char delimiter){
	char format_str[] ="%d,%s";
	format_str[2]=delimiter;
	std::string line;
	while (listhandler->next(&line)) {
		int oc;
		if (line.size() > 100) {
			line = line.substr(0, 100);
		}
		char* linebuf = new char[line.length()+5];
#ifdef _WIN32
		sscanf_s(line.c_str(), format_str, &oc, linebuf, line.length()+5);
#else
		sscanf(line.c_str(), format_str, &oc, linebuf);
#endif
		this->AdjustEdge((const char*)linebuf, oc); 
		delete linebuf;
	}
}


std::ofstream* MarkovPasswords::Save(const char* filename) {
	std::ofstream* exportFile;

	std::ofstream newFile(filename);

	exportFile = &newFile;
	
	this->Export(exportFile);
	return exportFile;
}


void MarkovPasswords::Generate(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads)  {
	char* res;
	char print[100];
	std::ofstream wordlist;	
	wordlist.open(wordlistFileName);
	std::mutex mlock;
	int iterationsPerThread = n/threads;
	int iterationsCarryOver = n%threads;
	std::vector<std::thread*> threadsV;
	for(int i=0;i<threads;i++){
		threadsV.push_back(new std::thread(&MarkovPasswords::GenerateThread, this, &mlock, iterationsPerThread, &wordlist, minLen, maxLen));
	}

	for(int i=0;i<threads;i++){
		threadsV[i]->join();
		delete threadsV[i];
	}

	this->GenerateThread(&mlock, iterationsCarryOver, &wordlist, minLen, maxLen);
	
}

void MarkovPasswords::GenerateThread(std::mutex *outputLock, unsigned long int n, std::ofstream *wordlist, int minLen, int maxLen)  {
	char* res = new char[64];
	if(n==0) return;

	Markov::Random::Marsaglia MarsagliaRandomEngine;
	for (int i = 0; i < n; i++) {
		this->RandomWalk(&MarsagliaRandomEngine, minLen, maxLen, res); 
		outputLock->lock();
		*wordlist << res << "\n";
		outputLock->unlock();
		//delete res;
	}
}