import os
HTML_EXTRA_FILES = "../README.md"

DOXYFILE = 'Doxyfile-mcss'
MAIN_PROJECT_URL = "https://github.com/FlameOfIgnis/CNG491-Project"

SHOW_UNDOCUMENTED = True
HTML_HEADER = open('documentation/includes/header.html').read()


x = ["images/" + (item) for item in os.listdir('documentation/images/')]
print(x)
HTML_EXTRA_FILES = x

FINE_PRINT = """CNG 491 - 2020  <br>
Ata Hakçıl <br>
Ömer Yıldıztugay <br>
Celal Sahir Çetiner <br>
Yunus Emre Yılmaz"""
