/** @file Generate.cpp
 * @brief Generation Page
 * @authors Yunus Emre Yılmaz
 *
 */

#include "Generate.h"
#include <fstream>
#include<QFileDialog>
#include<QMessageBox>
#include<QTextStream>
#include<QDir>
#include "CLI.h"
#include "MarkovAPI/src/markovPasswords.h"
#include <QtWidgets/QApplication>
#include "MarkovPasswordsGUI.h"
using namespace Markov::GUI;


Generate::Generate(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {generation(); });
    QObject::connect(ui.pushButton_4, &QPushButton::clicked, this, [this] {home(); });
    QObject::connect(ui.pushButton_3, &QPushButton::clicked, this, [this] {train(); });
    QObject::connect(ui.pushButton_2, &QPushButton::clicked, this, [this] {vis(); });


    ui.pushButton->setVisible(false);
    ui.lineEdit->setVisible(false);
    ui.lineEdit_2->setVisible(false);
    ui.lineEdit_3->setVisible(false);
    ui.label_3->setVisible(false);
    ui.label_4->setVisible(false);
    ui.label_5->setVisible(false);


}

void Generate::generation() {

   


    QString file_name = QFileDialog::getOpenFileName(this, "Open a File", QDir::homePath());
    QFile file(file_name);

    

    int numberPass = ui.lineEdit->text().toInt();
    int minLen = ui.lineEdit_2->text().toInt();
    int maxLen = ui.lineEdit_3->text().toInt();
    char* cstr;
    std::string fname = file_name.toStdString();
    cstr = new char[fname.size() + 1];
    strcpy(cstr, fname.c_str());
    
    ui.label_6->setText("GENERATING!");
    
    Markov::API::MarkovPasswords mp;
    mp.Import("src\\CLI\\sample_models\\2gram-trained.mdl");

    mp.Generate(numberPass,cstr,minLen,maxLen);

    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, "Error", "File Not Open!");
    }
    QTextStream in(&file);
    QString text = in.readAll();
    ui.plainTextEdit->setPlainText(text);
    
    ui.label_6->setText("DONE!");
    
    
   
    file.close();
}


void Generate::train() {
    QString file_name = QFileDialog::getOpenFileName(this, "Open a File", QDir::homePath());
    QFile file(file_name);

    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, "Error", "File Not Open!");
    }
    QTextStream in(&file);
    QString text = in.readAll();


    char* cstr;
    std::string fname = file_name.toStdString();
    cstr = new char[fname.size() + 1];
    strcpy(cstr, fname.c_str());



    char a = ',';
    Markov::API::MarkovPasswords mp;
    mp.Import("models\\2gram.mdl");
    mp.Train(cstr, a,10);
    mp.Export("models\\finished.mdl");



    ui.pushButton->setVisible(true);
    ui.lineEdit->setVisible(true);
    ui.lineEdit_2->setVisible(true);
    ui.lineEdit_3->setVisible(true);
    ui.label_3->setVisible(true);
    ui.label_4->setVisible(true);
    ui.label_5->setVisible(true);

    file.close();


}

void Generate::home() {
     CLI* w = new CLI;
     w->show();
     this->close();
}
void Generate :: vis() {
    MarkovPasswordsGUI* w = new MarkovPasswordsGUI;
    w->show();
    this->close();
}