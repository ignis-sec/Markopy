/** @file MarkovPasswordsGUI.h
 * @brief Main activity page for GUI
 * @authors Yunus Emre Yılmaz
 *
 */

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

        void benchmarkSelected();
        //void MarkovPasswordsGUI::modelvisSelected();
        //void MarkovPasswordsGUI::visualDebugSelected();
        //void MarkovPasswordsGUI::comparisonSelected();
    
   
  public slots:

      void home();
      void pass();
      void model();
  };
};


