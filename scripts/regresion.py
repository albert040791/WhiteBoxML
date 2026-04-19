from __future__ import annotations
import numpy as np

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegression:
    """
    Regresión logística simple.

    :param learning_rate: velocidad de aprendizaje
    :param n_iters: cantidad de iteraciones
    :authors: Albert José Sanchez Almarat
    :date: 12/04/2026
    """

    def __init__(self, learning_rate: float = 0.1, n_iters: int = 1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Función sigmoide.

        :param z: valor
        :return: resultado entre 0 y 1
        :authors: Albert José Sanchez Almarat
        :date: 12/04/2026
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo.

        :param X: datos
        :param y: etiquetas
        :return: None
        :authors: Albert José Sanchez Almarat
        :date: 12/04/2026
        """
        n, m = X.shape

        self.w = np.zeros(m)

        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)

            error = y_pred - y

            self.w = self.w - self.lr * (X.T @ error) / n
            self.b = self.b - self.lr * np.sum(error) / n

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice 0 o 1.

        :param X: datos
        :return: predicciones
        :authors: Albert José Sanchez Almarat
        :date: 12/04/2026
        """
        z = X @ self.w + self.b
        y_pred = self.sigmoid(z)

        return (y_pred >= 0.5).astype(int)
    




def test_basico():
    """
    Test simple.
    """
    X = np.array([[1], [2], [3], [10], [11], [12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    pred = model.predict(X)

    assert (pred == y).mean() > 0.8