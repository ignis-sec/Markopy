#include "MarkovPasswordsGUI.h"
#include <fstream>
#include <qwebengineview.h>
#include <Windows.h>
#include <QtWidgets/QApplication>


MarkovPasswordsGUI::MarkovPasswordsGUI(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    
    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {benchmarkSelected(); });
    QObject::connect(ui.pushButton_2,&QPushButton::clicked, this, [this] {modelvisSelected(); });
    QObject::connect(ui.pushButton_4, &QPushButton::clicked, this, [this] {comparisonSelected(); });
}


/*
Methods for buttons
*/

void MarkovPasswordsGUI::benchmarkSelected() {
   
    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\example.html";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));
}


void MarkovPasswordsGUI::modelvisSelected() {

    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\model.htm";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));
}

void MarkovPasswordsGUI::comparisonSelected() {

    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\comparison.htm";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));
}





void MarkovPasswordsGUI::renderHTMLFile(std::string* filename) {
    //extract and parametrize the code from constructor

}



void MarkovPasswordsGUI::loadDataset(std::string* filename) {
    //extract and parametrize the code from constructor

}