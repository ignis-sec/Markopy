#include "MarkovPasswordsGUI.h"
#include <fstream>
#include <qwebengineview.h>
#include <Windows.h>
#include "CLI.h"

using namespace Markov::GUI;

Markov::GUI::MarkovPasswordsGUI::MarkovPasswordsGUI(QWidget *parent) : QMainWindow(parent){
    ui.setupUi(this);


    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {home(); });
    QObject::connect(ui.pushButton_2, &QPushButton::clicked, this, [this] {model(); });
    QObject::connect(ui.pushButton_3, &QPushButton::clicked, this, [this] {pass(); });
}


void MarkovPasswordsGUI::home() {
    CLI* w = new CLI;
    w->show();
    this->close();
}
void MarkovPasswordsGUI::pass() {
    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\bar.html";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));
}

void MarkovPasswordsGUI::model() {
    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\index.html";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));
}