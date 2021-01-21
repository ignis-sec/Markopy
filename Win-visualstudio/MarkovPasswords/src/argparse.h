
//#include <boost/program_options.hpp>
/** @brief Structure to hold parsed cli arguements.
	
*/
typedef struct _programOptions {
	bool bImport;
	bool bExport;
	bool bFailure;
	char importname[256];
	char exportname[256];
	char outputfilename[256];
	char datasetname[256];
	int generateN;
} ProgramOptions;


/** @brief Parse command line arguements.
*/
static class Argparse {
public:
	/** @brief parse cli commands and return 
	* @param argc - Program arguement count
	* @param argv - Program arguement values array
	* @return ProgramOptions structure.
	*/
	static ProgramOptions* parse(int argc, char** argv);

	/** @brief Print help string.
	*/
	static void help();
private:

};