#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MarkovPasswordsGUI.h"



namespace Markov::GUI{
    /** @brief Reporting UI.
    *   
    * UI for reporting and debugging tools for MarkovPassword
    */
    class MarkovPasswordsGUI : public QMainWindow {
        Q_OBJECT
    public:
        MarkovPasswordsGUI(QWidget* parent = Q_NULLPTR);

    private:
        Ui::MarkovPasswordsGUIClass ui;


        //Slots for buttons in GUI.
    public slots:

        void MarkovPasswordsGUI::benchmarkSelected();
        //void MarkovPasswordsGUI::modelvisSelected();
        //void MarkovPasswordsGUI::visualDebugSelected();
        //void MarkovPasswordsGUI::comparisonSelected();
    
   
  public slots:

      void MarkovPasswordsGUI::home();
      void MarkovPasswordsGUI :: pass();
      void MarkovPasswordsGUI::model();
  };
};


