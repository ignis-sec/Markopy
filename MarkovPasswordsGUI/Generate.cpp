#include "Generate.h"
#include <fstream>
#include<QFileDialog>
#include<QMessageBox>
#include<QTextStream>
#include<QDir>
#include "src/CLI.h"
// #include "../MarkovPasswords/src/markovPasswords.cpp"
#include <QtWidgets/QApplication>

using namespace Markov::GUI;

Generate::Generate(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    QObject::connect(ui.pushButton, &QPushButton::clicked, this, [this] {generation(); });

}

void Generate::generation() {
    QString file_name = QFileDialog::getOpenFileName(this, "Open a File", QDir::homePath());
    QFile file(file_name);

    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, "Error", "File Not Open!");
    }
    QTextStream in(&file);
    QString text = in.readAll();
    ui.plainTextEdit->setPlainText(text);



    file.close();
}

void Generate::home() {
     CLI* w = new CLI;
     w->show();
     this->close();
}