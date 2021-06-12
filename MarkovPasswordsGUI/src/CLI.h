#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_CLI.h"

namespace Markov::GUI{
	/** @brief QT CLI Class
	*/ 
	class CLI :public QMainWindow {
		Q_OBJECT
	public:
		CLI(QWidget* parent = Q_NULLPTR);

	private:
		Ui::CLI ui;

	public slots:
		void start();
		void statistics();
		void about();
	};
};