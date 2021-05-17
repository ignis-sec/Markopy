#include "Start.h"
#include <fstream>
#include <Windows.h>

#include <QtWidgets/QApplication>
#include "Train.h"


Start::Start(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);


    
    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {start(); });
}

void Start::start() {
    Train* w = new Train;
    w->show();
    this->close();
}