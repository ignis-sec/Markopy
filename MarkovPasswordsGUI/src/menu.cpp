/** @file menu.cpp
 * @brief menu page
 * @authors Yunus Emre Yılmaz
 *
 */

#include "menu.h"
#include <fstream>
#include "MarkovPasswordsGUI.h"
#include <QtWidgets/QApplication>

using namespace Markov::GUI;

menu::menu(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    

    //QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {about(); });
    QObject::connect(ui.visu, &QPushButton::clicked, this, [this] {visualization(); });
}
void menu::about() {

   
}
void menu::visualization() {
    MarkovPasswordsGUI* w = new MarkovPasswordsGUI;
    w->show();
    this->close();
}
