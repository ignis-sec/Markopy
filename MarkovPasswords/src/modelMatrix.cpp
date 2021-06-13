#include "modelMatrix.h"
#include <map>
#include <cstring>
#include <thread>

Markov::API::ModelMatrix::ModelMatrix(){

}


void Markov::API::ModelMatrix::ConstructMatrix(){
    this->matrixSize = this->StarterNode()->edgesV.size() + 2;

    this->matrixIndex = new char[this->matrixSize];
    this->totalEdgeWeights = new long int[this->matrixSize];

    this->edgeMatrix = new char*[this->matrixSize];
    for(int i=0;i<this->matrixSize;i++){
        this->edgeMatrix[i] = new char[this->matrixSize];
    }
    this->valueMatrix = new long int*[this->matrixSize];
    for(int i=0;i<this->matrixSize;i++){
        this->valueMatrix[i] = new long int[this->matrixSize];
    }
    std::map< char, Node< char > * > *nodes;
    nodes = this->Nodes();
    int i=0;
    for (auto const& [repr, node] : *nodes){
        if(repr!=0) this->matrixIndex[i] = repr;
        else this->matrixIndex[i] = 199;
        this->totalEdgeWeights[i] = node->TotalEdgeWeights();
        for(int j=0;j<this->matrixSize;j++){
            char val = node->NodeValue();
            if(val < 0){
                for(int k=0;k<this->matrixSize;k++){
                    this->valueMatrix[i][k] = 0;
                    this->edgeMatrix[i][k] = 255;
                }
                break;
            }
            else if(node->NodeValue() == 0 && j>(this->matrixSize-3)){
                this->valueMatrix[i][j] = 0;
                this->edgeMatrix[i][j] = 255;
            }else if(j==(this->matrixSize-1)) {
                this->valueMatrix[i][j] = 0;
                this->edgeMatrix[i][j] = 255;
            }else{
                this->valueMatrix[i][j] = node->edgesV[j]->EdgeWeight();
                this->edgeMatrix[i][j]  = node->edgesV[j]->RightNode()->NodeValue();
            }

        }
        i++;
    }

    //this->DumpJSON();
}


void Markov::API::ModelMatrix::DumpJSON(){

    std::cout << "{\n   \"index\": \"";
    for(int i=0;i<this->matrixSize;i++){
        if(this->matrixIndex[i]=='"') std::cout << "\\\"";
        else if(this->matrixIndex[i]=='\\') std::cout << "\\\\";
        else if(this->matrixIndex[i]==0) std::cout << "\\\\x00";
        else if(i==0) std::cout << "\\\\xff";
        else if(this->matrixIndex[i]=='\n') std::cout << "\\n";
        else std::cout << this->matrixIndex[i];
    }
    std::cout << 
    "\",\n"
    "   \"edgemap\": {\n";

    for(int i=0;i<this->matrixSize;i++){
        if(this->matrixIndex[i]=='"') std::cout << "      \"\\\"\": [";
        else if(this->matrixIndex[i]=='\\') std::cout << "      \"\\\\\": [";
        else if(this->matrixIndex[i]==0) std::cout << "      \"\\\\x00\": [";
        else if(this->matrixIndex[i]<0) std::cout << "      \"\\\\xff\": [";
        else std::cout << "      \"" << this->matrixIndex[i] << "\": [";
        for(int j=0;j<this->matrixSize;j++){
            if(this->edgeMatrix[i][j]=='"') std::cout << "\"\\\"\"";
            else if(this->edgeMatrix[i][j]=='\\') std::cout << "\"\\\\\"";
            else if(this->edgeMatrix[i][j]==0) std::cout << "\"\\\\x00\"";
            else if(this->edgeMatrix[i][j]<0) std::cout << "\"\\\\xff\"";
            else if(this->matrixIndex[i]=='\n') std::cout << "\"\\n\"";
            else std::cout << "\"" << this->edgeMatrix[i][j] << "\"";
            if(j!=this->matrixSize-1) std::cout << ", ";
        }
        std::cout << "],\n";
    }
    std::cout << "},\n";

    std::cout << "\"   weightmap\": {\n";
    for(int i=0;i<this->matrixSize;i++){
        if(this->matrixIndex[i]=='"') std::cout << "      \"\\\"\": [";
        else if(this->matrixIndex[i]=='\\') std::cout << "      \"\\\\\": [";
        else if(this->matrixIndex[i]==0) std::cout << "      \"\\\\x00\": [";
        else if(this->matrixIndex[i]<0) std::cout << "      \"\\\\xff\": [";
        else std::cout << "      \"" << this->matrixIndex[i] << "\": [";

        for(int j=0;j<this->matrixSize;j++){
            std::cout << this->valueMatrix[i][j];
            if(j!=this->matrixSize-1) std::cout << ", ";
        }
        std::cout << "],\n";
    }
    std::cout << "  }\n}\n";
}


void Markov::API::ModelMatrix::FastRandomWalkThread(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, int id, bool bFileIO){
    if(n==0) return;

    Markov::Random::Marsaglia MarsagliaRandomEngine;
    char* e;
    char *res = new char[maxLen*n];
    int index = 0;
    char next;
    int len=0;
    long int selection;
    char cur;
    long int bufferctr = 0;
    for (int i = 0; i < n; i++) {
        cur=199;
        len=0;
        while (true) {
            e = strchr(this->matrixIndex, cur);
            index = e - this->matrixIndex;
            selection = MarsagliaRandomEngine.random() % this->totalEdgeWeights[index];
            for(int j=0;j<this->matrixSize;j++){
                selection -= this->valueMatrix[index][j];
                if (selection < 0){
                    next = this->edgeMatrix[index][j];
                    break;
                }
            }

            if (len >= maxLen)  break;
            else if ((next < 0) && (len < minLen)) continue;
            else if (next < 0) break;  
            cur = next;
            res[bufferctr + len++] = cur;
        }
        res[bufferctr + len++] = '\n';
        bufferctr+=len;
        
    }
    if(bFileIO){
        mlock->lock();
        *wordlist << res;
        mlock->unlock();
    }else{
        mlock->lock();
        std::cout << res;
        mlock->unlock();
    }
    delete res;

}


void Markov::API::ModelMatrix::FastRandomWalk(unsigned long int n, const char* wordlistFileName, int minLen, int maxLen, int threads, bool bFileIO){
    

    std::ofstream wordlist;	
    if(bFileIO)
        wordlist.open(wordlistFileName);

    std::mutex mlock;
    if(n<=50000000ull) return this->FastRandomWalkPartition(&mlock, &wordlist, n, minLen, maxLen, bFileIO, threads);
    else{
        int numberOfPartitions = n/50000000ull;
        for(int i=0;i<numberOfPartitions;i++)
            this->FastRandomWalkPartition(&mlock, &wordlist, 50000000ull, minLen, maxLen, bFileIO, threads);
    }


}


void Markov::API::ModelMatrix::FastRandomWalkPartition(std::mutex *mlock, std::ofstream *wordlist, unsigned long int n, int minLen, int maxLen, bool bFileIO, int threads){
    
    int iterationsPerThread = n/threads;
	int iterationsPerThreadCarryOver = n%threads;

	std::vector<std::thread*> threadsV;
    
    int id = 0;
	for(int i=0;i<threads;i++){
		threadsV.push_back(new std::thread(&Markov::API::ModelMatrix::FastRandomWalkThread, this, mlock, wordlist, iterationsPerThread, minLen, maxLen, id, bFileIO));
        id++;
	}

	threadsV.push_back(new std::thread(&Markov::API::ModelMatrix::FastRandomWalkThread, this, mlock, wordlist, iterationsPerThreadCarryOver, minLen, maxLen, id, bFileIO));

    for(int i=0;i<threads;i++){
		threadsV[i]->join();
	}
}