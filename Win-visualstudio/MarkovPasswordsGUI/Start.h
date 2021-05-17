#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_about.h"
#include <ui_Start.h>



class Start :public QMainWindow {
	Q_OBJECT
public:
	Start(QWidget* parent = Q_NULLPTR);

private:
	Ui::Start ui;

public slots:
	void start();
	
};