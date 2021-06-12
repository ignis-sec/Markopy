#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_menu.h"


namespace Markov::GUI{
	class menu:public QMainWindow {
	Q_OBJECT
	public:
		menu(QWidget* parent = Q_NULLPTR);

	private:
		Ui::main ui;

	public slots:
		void about();
		void visualization();
	};
};
