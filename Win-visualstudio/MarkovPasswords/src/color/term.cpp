#pragma once
#include "Term.h"


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


