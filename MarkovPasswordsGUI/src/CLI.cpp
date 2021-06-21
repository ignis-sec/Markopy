#include "CLI.h"
#include <fstream>
#include "Train.h"



using namespace Markov::GUI;

Markov::GUI::CLI::CLI(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.startButton, &QPushButton::clicked, this, [this] {start(); });
    QObject::connect(ui.commandLinkButton_2, &QPushButton::clicked, this, [this] {statistics(); });
    QObject::connect(ui.commandLinkButton, &QPushButton::clicked, this, [this] {about(); });
     
}

void Markov::GUI::CLI::start() {
    Train* w = new Train;
    w->show();
    this->close();
}
void Markov::GUI::CLI::statistics() {
    /*
    statistic will show
    */
}
void Markov::GUI::CLI::about() {
    /*
    about button
    */
}