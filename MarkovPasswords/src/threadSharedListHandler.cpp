#include "threadSharedListHandler.h"


ThreadSharedListHandler::ThreadSharedListHandler(const char* filename){
    this->listfile;
	this->listfile.open(filename, std::ios_base::binary);
}


bool ThreadSharedListHandler::next(std::string* line){
    bool res = false;
    this->mlock.lock();
    res = (std::getline(this->listfile,*line,'\n'))? true : false;
    this->mlock.unlock();
    
    return res;
}