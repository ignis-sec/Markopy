#pragma once
#include "markovPasswords.h"


MarkovPasswords::MarkovPasswords() : Markov::Model<char>(){
	
	
}

MarkovPasswords::MarkovPasswords(char* filename) {
	
	std::ifstream* importFile;

    this->Import(filename);
	
	//std::ifstream* newFile(filename);
	
	//importFile = newFile;	

}

std::ifstream* MarkovPasswords::OpenDatasetFile(char* filename){

	std::ifstream* datasetFile;

	std::ifstream newFile(filename);

	datasetFile = &newFile;

	this->Import(datasetFile);
	return datasetFile;
}


void MarkovPasswords::Train(std::ifstream* datasetFile)   {
	
	std::string line;
	
	std::cout << 0;
	while (std::getline(*datasetFile,line,'\n')) {
		int oc;
	    char pass[128];     
		if (line.size() > 100) {
			line.substr(0, 100);
		}
		sscanf_s(line.c_str(), "%d\x09%s", &oc, pass);
		this->adjust(pass, oc);
	}
	this->Export(exportFileName);
	
}


std::ofstream* MarkovPasswords::Save(char* filename) {
	std::ofstream* exportFile;

	std::ofstream newFile(filename);

	exportFile = &newFile;
	
	this->Export(exportFile);
	return exportFile;
}


void MarkovPasswords::Generate(unsigned long int n, char* wordlistFileName)  {
	char* res;
	char print[100];
	std::ofstream wordlist;	

	
	wordlist.open(wordlistFileName);
	for (int i = 0; i < n; i++) {
		this->RandomWalk();
#ifndef _WIN32
		strcpy_s(print, 100, (char*)res);

#else
		strcpy(print, (char*)res);
#endif // !_WIN32
		wordlist << res << "\n";
		delete res;
	}
}