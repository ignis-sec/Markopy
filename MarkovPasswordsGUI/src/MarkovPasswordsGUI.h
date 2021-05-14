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

    /** @brief Render a HTML file.
    * @param filename - Filename of the html file. (relative path to the views folder).
    */
    void renderHTMLFile(std::string* filename);

    /** @brief Load a dataset to current view..
    * @param filename - Filename of the dataset file. (relative path to the views folder).
    */
    void loadDataset(std::string* filename);

private:
    Ui::MarkovPasswordsGUIClass ui;


    //Slots for buttons in GUI.
public slots:

    void MarkovPasswordsGUI::benchmarkSelected();
    void MarkovPasswordsGUI::modelvisSelected();
    void MarkovPasswordsGUI::visualDebugSelected();
    void MarkovPasswordsGUI::comparisonSelected();
};

