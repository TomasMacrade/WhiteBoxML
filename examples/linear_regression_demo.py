# -*- coding: utf-8 -*-
"""
Script de demostración para el modelo de Regresión Lineal.
Compara Solución Analítica vs Diferentes modos de Gradiente.

:authors: Equipo Regresión Lineal
:date: 10/05/26       
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import time  # Para medir eficiencia
from sklearn.datasets import fetch_california_housing
from whiteboxml.linear_models.linear_regression import LinearRegression

def run_demo():
    print("--- WhiteBoxML: Análisis Comparativo de Regresión Lineal ---")
    
    data = fetch_california_housing()
    X, y = data.data, data.target

    modelos = {
        "Analítico": LinearRegression(method="analytic"),
        "GD Batch": LinearRegression(method="gd", mode="batch", iterations=1000, normalize=True),
        "Mini-Batch": LinearRegression(method="gd", mode="mbgd", batch_size=32, iterations=1000, normalize=True),
        "SGD": LinearRegression(method="gd", mode="sgd", iterations=1000, normalize=True)
    }

    resultados = {}

    for nombre, model in modelos.items():
        start = time.time()
        model.fit(X, y)
        end = time.time()
        
        costo_final = model.compute_cost(model._add_intercept(model.fit_standardize(X)[0] if model.normalize else X), y, model._theta_scaled)
        resultados[nombre] = {"tiempo": end - start, "costo": costo_final, "historial": model.cost_history}
        print(f"✓ {nombre} entrenado.")

    plt.figure(figsize=(12, 7))
    for nombre in ["GD Batch", "Mini-Batch", "SGD"]:
        plt.plot(resultados[nombre]["historial"], label=f"{nombre} (Final MSE: {resultados[nombre]['costo']:.4f})")

    plt.title("Convergencia de Métodos Iterativos en California Housing", fontsize=14)
    plt.xlabel("Iteraciones", fontsize=12)
    plt.ylabel("Costo (MSE) - Escala Logarítmica", fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    print("\n--- RESUMEN DE RENDIMIENTO ---")
    print(f"{'Método':<15} | {'Error Final (MSE)':<18} | {'Tiempo (s)':<10}")
    print("-" * 50)
    for m in resultados:
        print(f"{m:<15} | {resultados[m]['costo']:<18.6f} | {resultados[m]['tiempo']:<10.6f}")

    plt.show()

if __name__ == "__main__":
    run_demo()