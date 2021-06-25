
/** @file about.cpp
 * @brief About page 
 * @authors Yunus Emre Yılmaz
 *
 */

#include <fstream>
#include <QtWidgets/QApplication>
#include "about.h"

using namespace Markov::GUI;

about::about(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

}

