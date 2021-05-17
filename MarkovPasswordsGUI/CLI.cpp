#include "src\CLI.h"
#include <fstream>
#include <Windows.h>
#include "src/Train.h"





CLI::CLI(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.startButton, &QPushButton::clicked, this, [this] {start(); });
    QObject::connect(ui.commandLinkButton_2, &QPushButton::clicked, this, [this] {statistics(); });
    QObject::connect(ui.commandLinkButton, &QPushButton::clicked, this, [this] {about(); });

}

void CLI::start() {
    Train* w = new Train;
    w->show();
    this->close();
}
void CLI::statistics() {
    /*
    statistic will show
    */
}
void CLI::about() {
    /*
    about button
    */
}