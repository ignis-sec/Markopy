#pragma once
#include "markovPasswords.h"
#include <string.h>

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


void MarkovPasswords::Train(const char* datasetFileName, char delimiter)   {
	std::ifstream datasetFile;
	datasetFile.open(datasetFileName, std::ios_base::binary);
	std::string line;
	//std::string pass;
	char format_str[] ="%d,%s";
	format_str[2]=delimiter;
	while (std::getline(datasetFile,line,'\n')) {
		int oc;
		std::string pass;  //fixed
	    //char pass[512];  //caused segfault
		if (line.size() > 100) {
			line.substr(0, 100);
		}
#ifdef _WIN32
		sscanf_s(line.c_str(), format_str, &oc, pass);
#else
		sscanf(line.c_str(), format_str, &oc, pass);
#endif
		//std::cout << "parsed: "<<pass << "," << oc << "\n";
		this->AdjustEdge(pass.c_str(), oc); 
	}
	
}


std::ofstream* MarkovPasswords::Save(const char* filename) {
	std::ofstream* exportFile;

	std::ofstream newFile(filename);

	exportFile = &newFile;
	
	this->Export(exportFile);
	return exportFile;
}


void MarkovPasswords::Generate(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen)  {
	char* res;
	char print[100];
	std::ofstream wordlist;	

	
	wordlist.open(wordlistFileName);
	for (int i = 0; i < n; i++) {
		res = this->RandomWalk(minLen, maxLen); 
#ifdef _WIN32
		strcpy_s(print, 100, (char*)res);
#else
		strcpy(print, (char*)res);
#endif // !_WIN32

		wordlist << res << "\n";
		delete res;
	}
}

