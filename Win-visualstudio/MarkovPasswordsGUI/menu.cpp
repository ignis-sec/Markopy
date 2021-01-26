#include "menu.h"
#include <fstream>
#include <Windows.h>


menu::menu(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);


    //QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {about(); });
    QObject::connect(ui.visu, &QPushButton::clicked, this, [this] {visualization(); });
}
void menu::about() {

   
}
void menu::visualization() {


}