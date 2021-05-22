#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_Generate.h"


class Generate :public QMainWindow {
	Q_OBJECT
public:
	Generate(QWidget* parent = Q_NULLPTR);

private:
	Ui::Generate ui;

public slots:
	void home();
	void generation();
	void train();
};
