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

- [tcl-tk](https://formulae.brew.sh/formula/tcl-tk#default)
```bash
brew install tcl-tk
```

- [pyenv](https://formulae.brew.sh/formula/pyenv#default)
```bash
brew install pyenv
```

- [pycharm-ce](https://formulae.brew.sh/cask/pycharm-ce#default)

```bash
brew install --cask pycharm-ce
```

## Instalar Python
- Seguir pasos (A a E) para macos y zsh: https://github.com/pyenv/pyenv#b-set-up-your-shell-environment-for-pyenv
- Agregar el alias opcional del paso A a ~/.zshrc.
- Instalar la versión de tcl-tk mencionada más arriba. No instalar la versión 8 que menciona la guía.

```bash
pyenv install 3.12
pyenv global 3.12
```

> [!NOTE]
> Hacemos uso de esta versión de python por ser la que tiene mejor compatibilidad con los paquetes al momento
> del desarrollo de esta aplicación.

En caso de no poder hacer el link de python, ejecutar los siguientes comandos:
```bash
brew cleanup
brew doctor
```
Esto elimina symlinks que estén rotos y revisa si hay problemas.


## Paquetes pip

> [!NOTE]
> Recomiendo utilizar la versión de pip que viene incluída en la versión de python3 (pip3).

### pip comandos

- Crear ambiente virtual:
```bash
python -m venv .venv
```

- Activar ambiente virtual:
```bash
source .venv/bin/activate
echo $VIRTUAL_ENV
```

- Instalar paquetes de requirements.txt:

```bash
pip install -r requirements.txt
```

- Desinstalar paquetes de requirements.txt:

```bash
pip uninstall -r requirements.txt -y
```

- Extraer paquetes instalados a requirements.txt:

```bash
pip freeze > requirements.txt
```

# :speech_balloon: Autor <a name = "author"></a>

- [@andresbiso](https://github.com/andresbiso)

# :tada: Reconocimientos <a name = "acknowledgement"></a>

- https://github.com/github/gitignore
- https://gist.github.com/rxaviers/7360908 -> github emojis
- https://gist.github.com/Myndex/5140d6fe98519bb15c503c490e713233 -> github flavored markdown cheat sheet
