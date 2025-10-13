# ----- Libraries -----
import numpy as np
import sympy

# Class definition
class StressState:
    def __init__(self, stress_tensor):
        self.stress_tensor = stress_tensor

    def hydrostatic_stress(self):
        return np.trace(self.stress_tensor) / -3

    def deviatoric_tensor(self):
        diagonal_hydrostatic_tensor = np.diagflat(np.full((1, 3), self.hydrostatic_stress()))
        return self.stress_tensor - diagonal_hydrostatic_tensor

    def deviatoric_principal_stresses(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.deviatoric_tensor())

        return eigenvalues

    def deviatoric_principal_tensor(self):
        diagonal_matrix = np.diag(self.deviatoric_principal_stresses())

        return diagonal_matrix

    def principal_stresses(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.stress_tensor)

        return eigenvalues

    def principal_tensor(self):
        diagonal_matrix = np.diag(self.principal_stresses())

        return diagonal_matrix


test_array = np.array([[50, 5, 20],
                       [5, 50, 10],
                       [20, 10, 0]])
test = StressState(test_array)
print(test.deviatoric_tensor())
print(test.deviatoric_principal_tensor())
