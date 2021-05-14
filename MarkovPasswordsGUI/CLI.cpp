#include "CLI.h"
#include <fstream>
#include <Windows.h>
//#include "Start.h"



CLI::CLI(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.startButton, &QPushButton::clicked, this, [this] {start(); });
    QObject::connect(ui.commandLinkButton_2, &QPushButton::clicked, this, [this] {statistics(); });
    QObject::connect(ui.commandLinkButton, &QPushButton::clicked, this, [this] {about(); });

}

void CLI::start() {
    // Start* w = new Start;
     //w->show();
     //this->close();
}
void CLI::statistics() {

}
void CLI::about() {

}