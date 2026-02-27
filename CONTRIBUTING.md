# Contribuir a WhiteBoxML

Gracias por sumarte a WhiteBoxML.

Este proyecto prioriza rigor matemático, claridad conceptual y buenas prácticas de ingeniería. Para mantener coherencia y calidad, todas las contribuciones deben seguir estas pautas.

---

# 1. Reglas generales

- Todo Pull Request (PR) debe incluir tests.
- Todo código público debe tener docstrings completos en formato Sphinx/reST.
- Se deben usar anotaciones de tipo (`typing`) y tipos de retorno con `->`.
- El código debe pasar los hooks de pre-commit.
- No se aceptan implementaciones sin documentación ni pruebas.
- Evitar dependencias innecesarias.

---

# 2. Tipado obligatorio

Usar anotaciones de tipo según PEP 484.

Ejemplo:

```python
from __future__ import annotations
import numpy as np
from typing import Optional

def ejemplo(X: np.ndarray, y: np.ndarray, normalize: bool = True) -> np.ndarray:
```

## Reglas:

- Siempre indicar tipo de retorno con ->.

- Usar **Optional** cuando corresponda.

- Evitar **Any** salvo justificación fuerte.

# 3. Formato obligatorio de docstrings

Usamos estilo **Sphinx/reST** con *:param:* y *:type:* explícitos.

Plantilla mínima obligatoria:
```python
def funcion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Breve descripción de la función.

    :param a: Descripción del parámetro a.
    :type a: numpy.ndarray
    :param b: Descripción del parámetro b.
    :type b: numpy.ndarray
    :return: Descripción del retorno.
    :authors: Tomás Macrade
    :date: 27/02/2026
    """
```

## Obligatorio incluir:

- *: param :* para cada parámetro

- *: type :* para cada parámetro

- *: return :*

- *: authors :* para cada autor del proyecto

- *: date :*


## Opcional :

- : notes :

- : references :

# 4. Estándar para modelos

Todo modelo público debe implementar:

- *model = Modelo()*
- *model.fit(X, y)*
- *pred = model.predict(X)*

5. Tests obligatorios (pytest)

**Usar pytest.**

- Los tests deben ubicarse en tests/test_*.py.

- Los tests deben tener una descripción en su docstring, pero no deben llevar parámetros configurables.

Ejemplo:

```python 
import numpy as np
from numpy.testing import assert_allclose
from whiteboxml.linear_models import OLS


def test_ols_recupera_coeficientes():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta

    model = OLS()
    model.fit(X, y)

    assert_allclose(model.coef_, beta, rtol=1e-4)
```


# 6. Pre-commit y calidad de código

PRs con fallos en CI no serán mergeados.

# 7. Buenas prácticas de implementación

- Preferir operaciones vectorizadas sobre loops en Python.

- Seguir principios de código limpio.

# 8. Convención de ramas y commits

**Ramas:**

- feature/<area>-<descripcion>

- fix/<area>-<descripcion>

Ejemplo:

- feature/linear_models-ols

**Mensajes de commit:**

- linear_models: agrega OLS por ecuación normal


9. Criterios de aceptación

**Un PR será mergeado cuando:**

- Tests pasan.

- Docstrings completos.

- API consistente.

- Revisión aprobada por al menos un maintainer.

# 10. Filosofía

**WhiteBoxML prioriza:**

- Comprensión sobre automatización.

- Rigor sobre velocidad apresurada.

- Ciencia colaborativa sobre código individual.

Cada contribución es parte de un esfuerzo colectivo. La calidad es responsabilidad compartida.

Gracias por construir WhiteBoxML.

---
