#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_about.h"
#include <ui_menu.h>

/** @brief namespace for MarkovPasswords API GUI wrapper
*/ 
namespace Markov::GUI{

	/** @brief QT Class for about page
	*/ 
	class about :public QMainWindow {
	Q_OBJECT
	public:
		about(QWidget* parent = Q_NULLPTR);

	private:
		Ui:: main ui;

		
	};
};


