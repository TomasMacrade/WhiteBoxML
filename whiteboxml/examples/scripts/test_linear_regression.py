# examples/test_linear_regression.py

#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from whiteboxml.linear_models.linear_regression import LinearRegression

def main():
    # Datos simples (relación perfecta)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    print("x=:", X)
    print("y=:", y)

    model = LinearRegression(lr=0.01, n_iter=1000)

    model.fit(X, y)

    print("Parametros aprendidos:")
    print("w:", model.params["weights"])
    print("b:", model.params["bias"])

    preds = model.predict(X)
    print("Predicciones:", preds)

if __name__ == "__main__":
    main()