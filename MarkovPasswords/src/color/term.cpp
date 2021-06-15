#pragma once
#include "term.h"
#include <string>

using namespace Markov::API::CLI;

//Windows text processing is different from unix systems, so use windows header and text attributes
#ifdef _WIN32

HANDLE Terminal::_stdout;
HANDLE Terminal::_stderr;

std::map<Terminal::color, DWORD> Terminal::colormap = {
	{Terminal::color::BLACK, 0},
	{Terminal::color::BLUE, 1},
	{Terminal::color::GREEN, 2},
	{Terminal::color::CYAN, 3},
	{Terminal::color::RED, 4},
	{Terminal::color::MAGENTA, 5},
	{Terminal::color::BROWN, 6},
	{Terminal::color::LIGHTGRAY, 7},
	{Terminal::color::DARKGRAY, 8},
	{Terminal::color::YELLOW, 14},
	{Terminal::color::WHITE, 15},
	{Terminal::color::RESET, 15},
};


Terminal::Terminal() {
	Terminal::_stdout = GetStdHandle(STD_OUTPUT_HANDLE);
	Terminal::_stderr = GetStdHandle(STD_ERROR_HANDLE);
}

std::ostream& operator<<(std::ostream& os, const Terminal::color& c) {
	SetConsoleTextAttribute(Terminal::_stdout, Terminal::colormap.find(c)->second);
	return os;
}

#else

std::map<Terminal::color, int> Terminal::colormap = {
	{Terminal::color::BLACK, 30},
	{Terminal::color::BLUE, 34},
	{Terminal::color::GREEN, 32},
	{Terminal::color::CYAN, 36},
	{Terminal::color::RED, 31},
	{Terminal::color::MAGENTA, 35},
	{Terminal::color::BROWN, 0},
	{Terminal::color::LIGHTGRAY, 0},
	{Terminal::color::DARKGRAY, 0},
	{Terminal::color::YELLOW, 33},
	{Terminal::color::WHITE, 37},
	{Terminal::color::RESET, 0},
};

Terminal::Terminal() {
	/*this->;*/
}

std::ostream& operator<<(std::ostream& os, const Terminal::color& c) {
	char buf[6];
	sprintf(buf,"%d",Terminal::colormap.find(c)->second);
	os << "\e[1;" << buf << "m";
	return os;
}




#endif

