/** @file main.cpp
 * @brief Entry point for GUI
 * @authors Yunus Emre Yılmaz
 *
 */

//#include "MarkovPasswordsGUI.h"
#include "menu.h"
#include <QtWidgets/QApplication>
#include <QSplashScreen>
#include < QDateTime > 
#include "CLI.h"

using namespace Markov::GUI;

/** @brief Launch UI.
*/
int main(int argc, char *argv[])
{

  

    QApplication a(argc, argv);

    QPixmap loadingPix("views/startup.jpg");
    QSplashScreen splash(loadingPix);
    splash.show();
    QDateTime time = QDateTime::currentDateTime();
    QDateTime currentTime = QDateTime::currentDateTime();   //Record current time
    while (time.secsTo(currentTime) <= 5)                   //5 is the number of seconds to delay
    {
        currentTime = QDateTime::currentDateTime();
        a.processEvents();
    };

    
    CLI w;
    w.show();
    splash.finish(&w);
    return a.exec();
}

