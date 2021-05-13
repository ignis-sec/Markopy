#pragma once
#include "term.h"
#include <string>


//Windows text processing is different from unix systems, so use windows header and text attributes
#ifdef _WIN32

HANDLE terminal::_stdout;
HANDLE terminal::_stderr;

std::map<terminal::color, DWORD> terminal::colormap = {
	{terminal::color::BLACK, 0},
	{terminal::color::BLUE, 1},
	{terminal::color::GREEN, 2},
	{terminal::color::CYAN, 3},
	{terminal::color::RED, 4},
	{terminal::color::MAGENTA, 5},
	{terminal::color::BROWN, 6},
	{terminal::color::LIGHTGRAY, 7},
	{terminal::color::DARKGRAY, 8},
	{terminal::color::YELLOW, 14},
	{terminal::color::WHITE, 15},
	{terminal::color::RESET, 15},
};


terminal::terminal() {
	terminal::_stdout = GetStdHandle(STD_OUTPUT_HANDLE);
	terminal::_stderr = GetStdHandle(STD_ERROR_HANDLE);
}

std::ostream& operator<<(std::ostream& os, const terminal::color& c) {
	SetConsoleTextAttribute(terminal::_stdout, terminal::colormap.find(c)->second);
	return os;
}

#else

std::map<terminal::color, int> terminal::colormap = {
	{terminal::color::BLACK, 30},
	{terminal::color::BLUE, 34},
	{terminal::color::GREEN, 32},
	{terminal::color::CYAN, 36},
	{terminal::color::RED, 31},
	{terminal::color::MAGENTA, 35},
	{terminal::color::BROWN, 0},
	{terminal::color::LIGHTGRAY, 0},
	{terminal::color::DARKGRAY, 0},
	{terminal::color::YELLOW, 33},
	{terminal::color::WHITE, 37},
	{terminal::color::RESET, 0},
};

terminal::terminal() {
	/*this->;*/
}

std::ostream& operator<<(std::ostream& os, const terminal::color& c) {
	char buf[6];
	sprintf(buf,"%d",terminal::colormap.find(c)->second);
	os << "\e[1;" << buf << "m";
	return os;
}




#endif

