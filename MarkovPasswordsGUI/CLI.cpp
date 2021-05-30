#include "src\CLI.h"
#include <fstream>
#include "src/Train.h"
#include  "Generate.h"
#include "src/MarkovPasswordsGUI.h"

CLI::CLI(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.startButton, &QPushButton::clicked, this, [this] {start(); });
    QObject::connect(ui.commandLinkButton_2, &QPushButton::clicked, this, [this] {statistics(); });
    QObject::connect(ui.commandLinkButton, &QPushButton::clicked, this, [this] {about(); });

}

void CLI::start() {
    Generate* w = new Generate;
    w->show();
    this->close();
}
void CLI::statistics() {
    MarkovPasswordsGUI* w = new MarkovPasswordsGUI;
    w->show();
    this->close();
}
void CLI::about() {
    /*
    about button
    */
}