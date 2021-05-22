#include "MarkovPasswordsGUI.h"
#include <fstream>
#include <qwebengineview.h>
#include <Windows.h>
#include "CLI.h"


MarkovPasswordsGUI::MarkovPasswordsGUI(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);


    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {home(); });

    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\comparison.htm";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));

}


void MarkovPasswordsGUI::home() {
    CLI* w = new CLI;
    w->show();
    this->close();
}