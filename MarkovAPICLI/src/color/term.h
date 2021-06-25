#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif

#include <iostream>
#include <map>

#define TERM_FAIL "[" << Markov::API::CLI::Terminal::color::RED << "+" << Markov::API::CLI::Terminal::color::RESET << "] "
#define TERM_INFO "[" << Markov::API::CLI::Terminal::color::BLUE << "+" << Markov::API::CLI::Terminal::color::RESET << "] "
#define TERM_WARN "[" << Markov::API::CLI::Terminal::color::YELLOW << "+" << Markov::API::CLI::Terminal::color::RESET << "] "
#define TERM_SUCC "[" << Markov::API::CLI::Terminal::color::GREEN << "+" << Markov::API::CLI::Terminal::color::RESET << "] "

namespace Markov::API::CLI{
	/** @brief pretty colors for Terminal. Windows Only.
	*/
	class Terminal {
	public:

		/** Default constructor.
		* Get references to stdout and stderr handles.
		*/
		Terminal();

		enum color { RESET, BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, LIGHTGRAY, DARKGRAY, BROWN };
		#ifdef _WIN32
		static HANDLE _stdout;
		static HANDLE _stderr;
		static std::map<Markov::API::CLI::Terminal::color, DWORD> colormap;
		#else
		static std::map<Markov::API::CLI::Terminal::color, int> colormap;
		#endif
		
		
		
		static std::ostream endl;


	};

	/** overload for std::cout. 
	*/
	std::ostream& operator<<(std::ostream& os, const Markov::API::CLI::Terminal::color& c);

}

