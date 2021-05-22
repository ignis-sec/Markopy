#include <string>
#include <fstream>
#include <mutex>


class ThreadSharedListHandler{
public:    
    ThreadSharedListHandler(const char* filename);

    bool next(std::string* line);

private:
    std::ifstream listfile;
    std::mutex mlock;
};