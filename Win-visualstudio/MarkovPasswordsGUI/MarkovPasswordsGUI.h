#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MarkovPasswordsGUI.h"

class MarkovPasswordsGUI : public QMainWindow
{
    Q_OBJECT

public:
    MarkovPasswordsGUI(QWidget *parent = Q_NULLPTR);

private:
    Ui::MarkovPasswordsGUIClass ui;
};
