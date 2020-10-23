#pragma once

#include <Windows.h>
#include <iostream>
#include <map>

#define TERM_FAIL "[" << terminal::color::RED << "+" << terminal::color::RESET << "] "
#define TERM_INFO "[" << terminal::color::BLUE << "+" << terminal::color::RESET << "] "
#define TERM_WARN "[" << terminal::color::YELLOW << "+" << terminal::color::RESET << "] "
#define TERM_SUCC "[" << terminal::color::GREEN << "+" << terminal::color::RESET << "] "

class terminal {
public:

	terminal();
	static HANDLE _stdout;
	static HANDLE _stderr;
	const enum class color { RESET, BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, LIGHTGRAY, DARKGRAY, BROWN };
	static std::map<terminal::color, DWORD> colormap;
	static std::ostream endl;


};

std::ostream& operator<<(std::ostream& os, const terminal::color& c);
