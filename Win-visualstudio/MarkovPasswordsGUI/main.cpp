#include "MarkovPasswordsGUI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MarkovPasswordsGUI w;
    w.show();
    return a.exec();
}
