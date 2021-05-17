#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_Train.h"


class Train :public QMainWindow {
	Q_OBJECT
public:
	Train(QWidget* parent = Q_NULLPTR);

private:
	Ui::Train ui;

public slots:
	void open();
	void train();
};
