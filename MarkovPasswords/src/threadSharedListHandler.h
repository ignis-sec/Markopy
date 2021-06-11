#include <string>
#include <fstream>
#include <mutex>


/** @brief Simple class for managing shared access to file
 * 
 * This class maintains the handover of each line from a file to multiple threads.
 * 
 * When two different threads try to read from the same file while reading a line isn't completed, it can have unexpected results.
 * Line might be split, or might be read twice.
 * This class locks the read action on the list until a line is completed, and then proceeds with the handover.
 * 
*/
class ThreadSharedListHandler{
public:
    /** @brief Construct the Thread Handler with a filename
     * 
     * Simply open the file, and initialize the locks.
     * 
     * @b Example @b Use: Simple file read
     * @code{.cpp}
     * ThreadSharedListHandler listhandler("test.txt");
     * std::string line;
     * std::cout << listhandler->next(&line) << "\n";
     * @endcode
     * 
     * @b Example @b Use: Example use case from MarkovPasswords showing multithreaded access
     * @code{.cpp}
     *  void MarkovPasswords::Train(const char* datasetFileName, char delimiter, int threads)   {
     *       ThreadSharedListHandler listhandler(datasetFileName);
     *       auto start = std::chrono::high_resolution_clock::now();
     *  
     *       std::vector<std::thread*> threadsV;
     *       for(int i=0;i<threads;i++){
     *           threadsV.push_back(new std::thread(&MarkovPasswords::TrainThread, this, &listhandler, datasetFileName, delimiter));
     *       }
     * 
     *      for(int i=0;i<threads;i++){
     *           threadsV[i]->join();
     *           delete threadsV[i];
     *       }
     *       auto finish = std::chrono::high_resolution_clock::now();
     *       std::chrono::duration<double> elapsed = finish - start;
     *       std::cout << "Elapsed time: " << elapsed.count() << " s\n";
     * 
     *   }
     * 
     *   void MarkovPasswords::TrainThread(ThreadSharedListHandler *listhandler, const char* datasetFileName, char delimiter){
     *       char format_str[] ="%ld,%s";
     *       format_str[2]=delimiter;
     *       std::string line;
     *       while (listhandler->next(&line)) {
     *           long int oc;
     *           if (line.size() > 100) {
     *               line = line.substr(0, 100);
     *           }
     *           char* linebuf = new char[line.length()+5];
     *           sscanf_s(line.c_str(), format_str, &oc, linebuf, line.length()+5);
     *           this->AdjustEdge((const char*)linebuf, oc); 
     *           delete linebuf;
     *       }
     *   }
     * @endcode
     * 
     * @param filename Filename for the file to manage.
    */
    ThreadSharedListHandler(const char* filename);

    /** @brief Read the next line from the file.
     * 
     * This action will be blocked until another thread (if any) completes the read operation on the file.
     * 
     * @b Example @b Use: Simple file read
     * @code{.cpp}
     * ThreadSharedListHandler listhandler("test.txt");
     * std::string line;
     * std::cout << listhandler->next(&line) << "\n";
     * @endcode
     * 
    */
    bool next(std::string* line);

private:
    std::ifstream listfile;
    std::mutex mlock;
};