<p align="center">
    Trabajo Práctico Integrador - TDLC - UP
    <br>
    1C - 2025
    <br>
</p>

# :pencil: Table of Contents
- [Acerca De](#about)
- [Levantar Proyecto](#run_project)
- [Herramientas Utilizadas](#built_using)
- [Autor](#author)
- [Reconocimientos](#acknowledgement)

# :information_source: Acerca De <a name = "about"></a>
- Repositorio que contiene el trabajo práctico de la materia Teoría de la Computación y Lenguajes de la Universidad de Palermo.

# :wrench: Levantar Proyecto <a name = "run_project"></a>

1. Ver "Instalar paquetes de requirements.txt" en este mismo archivo.
2. Levantar una terminal.
3. Navegar hasta la carpeta del proyecto.
4. Activar el ambiente local:
```bash
tadp-venv\Scripts\activate
```
5. Levantar el proyecto:
```bash
python main.py
```

- En caso de que ya estemos utilizando otro ambiente de python podemos correr el siguiente comando para desactivarlo:
```bash
deactivate
```

# :hammer: Herramientas Utilizadas <a name = "built_using"></a>

Recomiendo utilizar [homebrew](https://brew.sh/) para instalar estos paquetes:

- [python@3.13](https://docs.brew.sh/Homebrew-and-Python#python-3)

```bash
brew install python@3.13
```
- [visual-studio-code](https://formulae.brew.sh/cask/visual-studio-code#default)
```bash
brew install --cask visual-studio-code
```

## Paquetes pip
Recomiendo utilizar la versión de pip que viene incluído en la versión de python3 (pip3) para instalar los siguientes paquetes:
- [pytest](https://pypi.org/project/pytest/)
```
pip install -Iv pytest==8.3.5
```

### pip comandos
- Instalar paquetes de requirements.txt:
```
python -m venv tdlc-venv
tadp-venv\Scripts\activate
pip install -r requirements.txt
```
- Desinstalar paquetes de requirements.txt:
```
pip uninstall -r requirements.txt -y
```
- Extraer paquetes instalados a requirements.txt:
```
pip freeze > requirements.txt
```

# :speech_balloon: Autor <a name = "author"></a>
- [@andresbiso](https://github.com/andresbiso)

# :tada: Reconocimientos <a name = "acknowledgement"></a>
- https://github.com/github/gitignore
- https://gist.github.com/rxaviers/7360908 -> github emojis
- https://gist.github.com/Myndex/5140d6fe98519bb15c503c490e713233 -> github flavored markdown cheat sheet
