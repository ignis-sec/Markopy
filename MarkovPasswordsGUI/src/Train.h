/** @file Train.h
 * @brief training page for GUI
 * @authors Yunus Emre Yılmaz
 *
 */

#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_Train.h"

namespace Markov::GUI{

	/** @brief QT Training page class
	*/ 
	class Train :public QMainWindow {
	Q_OBJECT
	public:
		Train(QWidget* parent = Q_NULLPTR);

	private:
		Ui::Train ui;

	public slots:
		void home();
		void train();
	};
};

