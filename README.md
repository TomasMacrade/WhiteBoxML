# WhiteBoxML

**Un espacio abierto para aprender y desarrollar machine learning desde cero, en comunidad.**

Este README explica, de forma práctica y breve, qué contiene este repositorio, cómo instalar el entorno de desarrollo (Poetry), cómo ejecutar los checks básicos (`pre-commit`) y los tests (`pytest`), y dónde mirar para contribuir.

> Nota: el repositorio contiene un conjunto de archivos principales como `pyproject.toml` (dependencias y configuración), `CONTRIBUTING.md`, `.pre-commit-config.yaml`, `CITATION.cff` y `MANIFIESTO.md`.

---

## Contenido del repositorio (resumen de archivos)

- **`pyproject.toml`**  
  Define el paquete, la versión y las dependencias de desarrollo e instalación (acá se usan `numpy`, `pandas` y `matplotlib` junto a herramientas de desarrollo como `pytest` y `ruff`).

- **`.pre-commit-config.yaml`**  
  Configuración de hooks locales que se ejecutan antes de cada commit. Asegura estilo y chequeos básicos antes de subir cambios.

- **`CONTRIBUTING.md`**  
  Guía con las reglas de contribución: docstrings en estilo Sphinx/reST, tests obligatorios en `tests/`, convenciones de ramas y commits, y criterios para que un PR sea mergeado. Leerlo antes de abrir PRs.

- **`MANIFIESTO.md`**  
  Documento formal con los principios del proyecto: colaboración, apertura, ciencia en equipo, transparencia y formación.

- **`CITATION.cff`**  
  Archivo para indicar cómo citar el proyecto (GitHub lo mostrará en “Cite this repository” ). La idea es que todos podamos ser reconocidos si el proyecto escala.

- **`LICENSE`**  
  Licencia MIT en la raíz: define que el proyecto es libre para usar, modificar y distribuir con la condición de mantener el aviso de copyright.

- **Carpetas esperadas**  
  - `whiteboxml/` — código fuente del paquete (módulos como `linear_models`, `optimization`, `utils`, etc.).  
  - `tests/` — tests de pytest (`tests/test_*.py`).  
  - `examples/` — scripts de ejemplo para ejecutar y entender uso básico.  
  - `sideproyects/` — aplicaciones interesantes de ML

---

## Instalación (rápida) — usando Poetry (recomendado)

1. **Clonar el repositorio**

```bash
git clone https://github.com/TomasMacrade/WhiteBoxML.git
cd WhiteBoxML
```

2. **Instalar Poetry (si no lo tenés)**

```bash
curl -sSL https://install.python-poetry.org | python3 -
# o seguí la guía oficial en https://python-poetry.org/
```

3. **Instalar dependencias del proyecto**

```bash
poetry install
```
Esto instalará numpy, pandas, matplotlib (dependencias de runtime) y las dependencias de desarrollo (pytest, pre-commit, ruff, etc.) según pyproject.toml.

4. **Ejecutar comandos**
   
```bash
poetry run pytest
poetry run python examples/linear_regression_example.py
```

5. **Pueden abrir notebooks de jupyter de la siguiente forma**:

```bash
poetry run jupyter notebook
```


## Pre-commit — configurar y usar

Instalar hooks (desde el entorno donde instalaste pre-commit):

```bash
poetry run pre-commit install
```

Ejecutar todos los hooks localmente (para verificar antes de un commit):

```bash
poetry run pre-commit run --all-files
```

pre-commit aplica formateo automático y chequeos (Black, isort, ruff, mypy si está configurado). Es la primera línea de defensa para mantener calidad. Revisá .pre-commit-config.yaml para ver exactamente qué hooks están activos.


## Tests (pytest)

Los tests están en tests/ y se ejecutan con:

```bash
poetry run pytest
```

**Recomendación antes de abrir PR:**  ejecutar pre-commit y pytest localmente para evitar fallos en CI. (La política del repo exige tests en cada PR; ver CONTRIBUTING.md).


## Cómo contribuir (resumen práctico)

- Leé CONTRIBUTING.md y MANIFIESTO.md para entender las reglas y la cultura del proyecto.

- Creá una rama con convención feature/<area>-<descripción> o fix/<area>-<descripción>.

- Implementá la funcionalidad con docstrings Sphinx/reST (:param:, :type:, :return:) y tipado cuando aplique.

- Incluí tests en tests/.

- Ejecutá pre-commit y pytest localmente.

- Abrí un Pull Request con descripción clara y checklist (tests incluidos, docstrings, pre-commit pasado)

