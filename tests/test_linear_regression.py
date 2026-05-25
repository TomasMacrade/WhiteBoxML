import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from numpy.testing import assert_allclose
from whiteboxml.linear_models import LinearRegression

def test_linear_regression_diabetes_fit():
    """
    Test general: Verifica que el modelo pueda entrenar con el dataset de Diabetes
    y generar predicciones coherentes.
    :authors: Equipo Regresión Lineal
    :date: 10/05/26  
    """

    diabetes = load_diabetes()
    X = diabetes.data[:, 2:3]  # BMI
    y = diabetes.target

    model = LinearRegression(
        method="gd", 
        learning_rate=0.1, 
        iterations=1000, 
        normalize=True
    )
    model.fit(X, y)

    assert model._trained is True
    assert len(model.theta) == 2
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

def test_linear_regression_all_gradient_modes():
    """
    Test de robustez: Verifica que los 3 modos de gradiente (batch, sgd, mbgd)
    converjan a una solución similar en un entorno controlado.
    :authors: Equipo Regresión Lineal
    :date: 10/05/26  
    """
   
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, (100, 1))
    y = 10 + 5 * X.flatten() + rng.normal(0, 0.01, 100)

    modos = ["batch", "sgd", "mbgd"]
    
    for modo in modos:
        model = LinearRegression(
            method="gd",
            mode=modo,
            learning_rate=0.01,
            iterations=3000,
            normalize=True,
            batch_size=10
        )
        model.fit(X, y)

       
        expected = np.array([10.0, 5.0])
        assert_allclose(
            model.theta, 
            expected, 
            atol=1.0, 
            err_msg=f"Fallo en modo: {modo}"
        )

def test_linear_regression_analytic_exact():
    """
    Test de precisión: Verifica que el método analítico sea exacto.
    :authors: Equipo Regresión Lineal
    :date: 10/05/26  
    """
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30]) # y = 0 + 10x

    model = LinearRegression(method="analytic", normalize=False)
    model.fit(X, y)

    expected = np.array([0.0, 10.0])
    assert_allclose(model.theta, expected, atol=1e-7)