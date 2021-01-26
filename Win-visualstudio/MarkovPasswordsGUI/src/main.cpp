//#include "MarkovPasswordsGUI.h"
#include "menu.h"
#include <QtWidgets/QApplication>


/** @brief Launch UI.
*/
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    menu w;
    w.show();
    return a.exec();
}
