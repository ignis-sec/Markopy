#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_menu.h"


class menu:public QMainWindow {
	Q_OBJECT


public:
	menu(QWidget* parent = Q_NULLPTR);

private:
	Ui::menu ui;

public slots:
	void about();
	void visualization();
};