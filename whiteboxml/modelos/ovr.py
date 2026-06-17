"""
Clasificación multiclase con estrategia One vs Rest (OvR).

Dado un clasificador binario con interfaz fit/predict_proba,
entrena K clasificadores (uno por clase) y predice la clase
con mayor probabilidad.

:authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
:date: 14/06/2026
"""

# pylint: disable=invalid-name

from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from whiteboxml.base_linear_model import BaseLinearModel


class OneVsRest:
    """
    Clasificador multiclase con estrategia One vs Rest.

    Entrena un clasificador binario por cada clase. En la predicción
    elige la clase cuyo clasificador devuelve mayor probabilidad.

    El modelo binario debe heredar de BaseLinearModel e implementar
    predict_proba(X: ndarray) -> ndarray de shape (n_samples,).

    :param model_class: clase del modelo binario (debe heredar de BaseLinearModel).
    :param model_kwargs: parámetros pasados al constructor del modelo.
    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """

    def __init__(self, model_class: type, **model_kwargs) -> None:
        """
        Inicializa OneVsRest.

        :param model_class: clase del modelo binario.
        :param model_kwargs: kwargs para el constructor del modelo.
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        self._validate_model_class(model_class)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.classes_: np.ndarray | None = None
        self.models_: list | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OneVsRest":
        """
        Entrena un clasificador binario por cada clase detectada en y.

        :param X: matriz de features de shape (n_samples, n_features).
        :param y: vector de etiquetas de shape (n_samples,).
        :return: instancia entrenada.
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.models_ = []

        for cls in self.classes_:
            y_bin = (y == cls).astype(int)
            model = self.model_class(**self.model_kwargs)
            model.fit(X, y_bin)
            self.models_.append(model)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Devuelve la probabilidad de pertenencia a cada clase.

        :param X: matriz de features de shape (n_samples, n_features).
        :return: matriz de shape (n_samples, n_clases).
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        self._check_fitted()
        models = cast(list, self.models_)
        X = np.asarray(X)
        return np.column_stack([m.predict_proba(X) for m in models])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predice la clase con mayor probabilidad para cada muestra.

        :param X: matriz de features de shape (n_samples, n_features).
        :return: vector de etiquetas predichas de shape (n_samples,).
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        classes = cast(np.ndarray, self.classes_)
        return classes[np.argmax(self.predict_proba(X), axis=1)]

    def _check_fitted(self) -> None:
        """
        Verifica que el modelo fue entrenado antes de predecir.

        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        if self.models_ is None or self.classes_ is None:
            raise RuntimeError("El modelo no fue entrenado. Llamá fit() primero.")

    @staticmethod
    def _validate_model_class(model_class: type) -> None:
        """
        Valida que model_class herede de BaseLinearModel e implemente predict_proba.

        :param model_class: clase a validar.
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        if not isinstance(model_class, type):
            raise TypeError("model_class debe ser una clase, no una instancia.")
        if not issubclass(model_class, BaseLinearModel):
            raise TypeError(f"{model_class.__name__} debe heredar de BaseLinearModel.")
        dummy = object.__new__(model_class)
        if not hasattr(dummy, "predict_proba"):
            raise TypeError(f"{model_class.__name__} debe implementar predict_proba().")
