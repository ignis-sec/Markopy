#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MarkovPasswordsGUI.h"

/** @brief Reporting UI.
*   
* UI for reporting and debugging tools for MarkovPassword
*/
class MarkovPasswordsGUI : public QMainWindow {
    Q_OBJECT

public:
    /** @brief Default QT consturctor.
    * @param parent - Parent widget.
    */
    MarkovPasswordsGUI(QWidget *parent = Q_NULLPTR);

  
private:
    Ui::MarkovPasswordsGUIClass ui;


   
public slots:

    void MarkovPasswordsGUI::home();
    void MarkovPasswordsGUI :: pass();
    void MarkovPasswordsGUI::model();
  
};

