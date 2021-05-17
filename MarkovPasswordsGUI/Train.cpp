#include "src/Train.h"
#include <fstream>
#include <Windows.h>




Train::Train(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {open(); });
    QObject::connect(ui.pushButton_2, &QPushButton::clicked, this, [this] {train(); });
   

}

void Train::open() {
    // Start* w = new Start;
     //w->show();
     //this->close();
}
void Train::train() {
    /*
    train codes;
    */
}
