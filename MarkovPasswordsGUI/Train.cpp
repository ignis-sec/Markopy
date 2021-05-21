#include "src/Train.h"
#include <fstream>
#include<QFileDialog>
#include<QMessageBox>
#include<QTextStream>
#include<QDir>
#include "src/CLI.h"
#include "MarkovPasswords/src/markovPasswords.h"

#include <QtWidgets/QApplication>

Train::Train(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

   

    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {train(); });
    QObject::connect(ui.pushButton_2, &QPushButton::clicked, this, [this] {home(); });

    

  
}

void Train::train() {

    

    QString file_name = QFileDialog::getOpenFileName(this, "Open a File", QDir::homePath());
    QFile file(file_name);

    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, "Error", "File Not Open!");
    }
    QTextStream in(&file);
    QString text = in.readAll();
    ui.plainTextEdit->setPlainText(text);

   
    char* cstr;
    std::string fname = file_name.toStdString();
    cstr = new char[fname.size() + 1];
    strcpy(cstr, fname.c_str());


    char a=',';
    MarkovPasswords mp;
    mp.Train(cstr, a, 10); //please parameterize this hardcoded 10 threads

    file.close();
}

void Train::home() {
     CLI* w = new CLI;
     w->show();
     this->close();
}

