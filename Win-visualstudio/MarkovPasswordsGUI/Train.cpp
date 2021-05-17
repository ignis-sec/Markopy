#include "Train.h"
#include <fstream>
#include <Windows.h>
#include <QtWidgets/QApplication>


Train::Train(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);


    QObject::connect(ui.pushButton_2, &QPushButton::clicked, this, [this] {train(); });
    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {open(); });
}
void Train::open() {
    /*
    this will open a file and read.
    */
}
void Train::train() {
    /*
    this will open a file and read.
    */
}