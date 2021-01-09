#pragma once

#include <Windows.h>
#include <iostream>
#include <map>

#define TERM_FAIL "[" << terminal::color::RED << "+" << terminal::color::RESET << "] "
#define TERM_INFO "[" << terminal::color::BLUE << "+" << terminal::color::RESET << "] "
#define TERM_WARN "[" << terminal::color::YELLOW << "+" << terminal::color::RESET << "] "
#define TERM_SUCC "[" << terminal::color::GREEN << "+" << terminal::color::RESET << "] "

/** @brief pretty colors for terminal. Windows Only.
*/
class terminal {
public:

	/** Default constructor.
	* Get references to stdout and stderr handles.
	*/
	terminal();
	static HANDLE _stdout;
	static HANDLE _stderr;
	const enum class color { RESET, BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, LIGHTGRAY, DARKGRAY, BROWN };
	static std::map<terminal::color, DWORD> colormap;
	static std::ostream endl;


};

/** overload for std::cout. 
*/
std::ostream& operator<<(std::ostream& os, const terminal::color& c);
