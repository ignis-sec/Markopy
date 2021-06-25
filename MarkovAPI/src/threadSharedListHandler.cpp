/** @file threadSharedListHandler.cpp
 * @brief Thread-safe wrapper for std::ifstream
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::API::Concurrency::ThreadSharedListHandler
 * 
 */

#include "threadSharedListHandler.h"


Markov::API::Concurrency::ThreadSharedListHandler::ThreadSharedListHandler(const char* filename){
    this->listfile;
	this->listfile.open(filename, std::ios_base::binary);
}


bool Markov::API::Concurrency::ThreadSharedListHandler::next(std::string* line){
    bool res = false;
    this->mlock.lock();
    res = (std::getline(this->listfile,*line,'\n'))? true : false;
    this->mlock.unlock();
    
    return res;
}