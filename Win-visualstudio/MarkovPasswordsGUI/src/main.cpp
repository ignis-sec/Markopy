#include "MarkovPasswordsGUI.h"
#include <QtWidgets/QApplication>


/** @brief Launch UI.
*/
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MarkovPasswordsGUI w;
    w.show();
    return a.exec();
}
