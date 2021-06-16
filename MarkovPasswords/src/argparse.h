#include<string>
#include<iostream>

//#define BOOST_ALL_DYN_LINK 1
#define Boost_USE_STATIC_LIBS 1

#include <boost/program_options.hpp>
/** @brief Structure to hold parsed cli arguements.	
*/
namespace opt = boost::program_options;

/**
	@brief Namespace for the CLI objects
*/
namespace Markov::API::CLI{

	/** @brief Structure to hold parsed cli arguements.	*/
	typedef struct _programOptions {
		/** 
			@brief Import flag to validate import	
		*/
		bool bImport;

		/** 
			@brief Export flag to validate export	
		*/
		bool bExport;

		/** 
			@brief Failure flag to validate succesfull running	
		*/
		bool bFailure;

		/** 
			@brief Seperator character to use with training data. (character between occurence and value)"	
		*/
		char seperator;

		/** 
			@brief Import name of our model	
		*/
		std::string importname;

		/** 
			@brief Import name of our given wordlist	
		*/
		std::string exportname;

		/** 
			@brief Import name of our given wordlist	
		*/
		std::string wordlistname;

		/** 
			@brief Output name of our generated password list	
		*/
		std::string outputfilename;

		/** 
			@brief The name of the given dataset	
		*/
		std::string datasetname;

		/** 
			@brief Number of passwords to be generated	
		*/
		int generateN;

	} ProgramOptions;


	/** @brief Parse command line arguements 
	*/
	class Argparse {
	public:
		
		Argparse();
		
		/** @brief Parse command line arguements.
		 * 
		 * Parses command line arguements to populate ProgramOptions structure.
		 * 
		 * @param argc Number of command line arguements
		 * @param argv Array of command line parameters
		*/
		Argparse(int argc, char** argv) {
			
			/*bool bImp;
			bool bExp;
			bool bFail;
			char sprt;
			std::string imports;
			std::string exports;
			std::string outputs;
			std::string datasets;
			int generateN;
			*/
			opt::options_description desc("Options");
			
			
			desc.add_options()
				("generate", "Generate strings with given parameters")
				("train", "Train model with given parameters")
				("combine", "Combine")
				("import", opt::value<std::string>(), "Import model file")
				("output", opt::value<std::string>(), "Output model file. This model will be exported when done. Will be ignored for generation mode")
				("dataset", opt::value<std::string>(), "Dataset file to read input from training. Will be ignored for generation mode")
				("seperator", opt::value<char>(), "Seperator character to use with training data. (character between occurence and value)")
				("wordlist", opt::value<std::string>(), "Wordlist file path to export generation results to. Will be ignored for training mode")
				("count", opt::value<int>(), "Number of lines to generate. Ignored in training mode")
				("verbosity", "Output verbosity")
				("help", "Option definitions");

				opt::variables_map vm;

				opt::store(opt::parse_command_line(argc, argv, desc), vm);

				opt::notify(vm);

				//std::cout << desc << std::endl;
				if (vm.count("help")) {
				std::cout << desc << std::endl;
				}
				
				if (vm.count("output") == 0) this->po.outputfilename = "NULL";
				else if (vm.count("output") == 1) {
					this->po.outputfilename = vm["output"].as<std::string>();
					this->po.bExport = true;
				}
				else {
					this->po.bFailure = true;
					std::cout << "UNIDENTIFIED INPUT" << std::endl;
					std::cout << desc << std::endl;
				}
				

				if (vm.count("dataset") == 0) this->po.datasetname = "NULL";
				else if (vm.count("dataset") == 1) {
					this->po.datasetname = vm["dataset"].as<std::string>();
				}
				else {
					this->po.bFailure = true;
					std::cout << "UNIDENTIFIED INPUT" << std::endl;
					std::cout << desc << std::endl;
				}


				if (vm.count("wordlist") == 0) this->po.wordlistname = "NULL";
				else if (vm.count("wordlist") == 1) {
					this->po.wordlistname = vm["wordlist"].as<std::string>();
				}
				else {
					this->po.bFailure = true;
					std::cout << "UNIDENTIFIED INPUT" << std::endl;
					std::cout << desc << std::endl;
				}

				if (vm.count("import") == 0) this->po.importname = "NULL";
				else if (vm.count("import") == 1) {
					this->po.importname = vm["import"].as<std::string>();
					this->po.bImport = true;
				}
				else {
					this->po.bFailure = true;
					std::cout << "UNIDENTIFIED INPUT" << std::endl;
					std::cout << desc << std::endl;
				}

				
				if (vm.count("count") == 0) this->po.generateN = 0;
				else if (vm.count("count") == 1) {
					this->po.generateN = vm["count"].as<int>();
				}
				else {
					this->po.bFailure = true;
					std::cout << "UNIDENTIFIED INPUT" << std::endl;
					std::cout << desc << std::endl;
				}
				
				/*std::cout << vm["output"].as<std::string>() << std::endl;
				std::cout << vm["dataset"].as<std::string>() << std::endl;
				std::cout << vm["wordlist"].as<std::string>() << std::endl;
				std::cout << vm["output"].as<std::string>() << std::endl;
				std::cout << vm["count"].as<int>() << std::endl;*/
				

				//else if (vm.count("train")) std::cout << "train oldu" << std::endl;
		}

		/** @brief Getter for command line options
		 * 
		 * Getter for ProgramOptions populated by the arguement parser
		 * @returns ProgramOptions structure.
		*/
		Markov::API::CLI::ProgramOptions getProgramOptions(void) {
			return this->po;
		}

		/** @brief Initialize program options structure.
		 * 
		 * @param i boolean, true if import operation is flagged
		 * @param e boolean, true if export operation is flagged
		 * @param bf boolean, true if there is something wrong with the command line parameters
		 * @param s seperator character for the import function
		 * @param iName import filename
		 * @param exName export filename
		 * @param oName output filename
		 * @param dName corpus filename
		 * @param n number of passwords to be generated
		 * 
		*/
		void setProgramOptions(bool i, bool e, bool bf, char s, std::string iName, std::string exName, std::string oName, std::string dName, int n) {
			this->po.bImport = i;
			this->po.bExport = e;
			this->po.seperator = s;
			this->po.bFailure = bf;
			this->po.generateN = n;
			this->po.importname = iName;
			this->po.exportname = exName;
			this->po.outputfilename = oName;
			this->po.datasetname = dName;

			/*strcpy_s(this->po.importname,256,iName);
			strcpy_s(this->po.exportname,256,exName);
			strcpy_s(this->po.outputfilename,256,oName);
			strcpy_s(this->po.datasetname,256,dName);*/
			
		}

		/** @brief parse cli commands and return  
		* @param argc - Program arguement count
		* @param argv - Program arguement values array
		* @return ProgramOptions structure.
		*/
		static Markov::API::CLI::ProgramOptions* parse(int argc, char** argv);
			
		
		/** @brief Print help string.
		*/
		static void help();

	private:
		/** 
			@brief ProgramOptions structure object
		*/

		Markov::API::CLI::ProgramOptions po;
	};

};

