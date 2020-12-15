#include "MarkovPasswordsGUI.h"
#include <fstream>
#include <qwebengineview.h>

#include <Windows.h>
MarkovPasswordsGUI::MarkovPasswordsGUI(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QWebEngineView* webkit = ui.centralWidget->findChild<QWebEngineView*>("chartArea");

    //get working directory
    char path[255];
    GetCurrentDirectoryA(255, path);

    //get absolute path to the layout html
    std::string layout = "file:///" + std::string(path) + "\\views\\example.html";
    std::replace(layout.begin(), layout.end(), '\\', '/');
    webkit->setUrl(QUrl(layout.c_str()));

}
