# examples/reporte_basico.py

import numpy as np
from whiteboxml.linear_models import LogisticRegression
from whiteboxml.reporting.model_reporting import ModelReporting

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

reporter = ModelReporting(X, y)
reporter.compare_models([model])
