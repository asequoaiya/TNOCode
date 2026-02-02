# ----- Import libraries -----
import numpy as np
import pandas as pd

# Power law material definition
class PowerLawMaterial:
    def __init__(self, elastic_modulus, yield_strength, hardening_exponent, dataframe_name):
        self.elastic_modulus = elastic_modulus
        self.yield_strength = yield_strength
        self.hardening_exponent = hardening_exponent
        self.dataframe_name = dataframe_name

    def elastic_behavior(self):
        yield_strain = self.yield_strength / self.elastic_modulus
        return yield_strain

    def plastic_behavior(self):
        plastic_strains = np.linspace(0, 1, 101)
        fraction = self.elastic_modulus * (self.elastic_behavior() + plastic_strains) / self.yield_strength
        yield_stresses = self.yield_strength * np.power(fraction, self.hardening_exponent)

        plastic_array = np.vstack((yield_stresses, plastic_strains)).transpose()
        plastic_dataframe = pd.DataFrame(plastic_array)
        plastic_dataframe.to_csv(f"{self.dataframe_name}.csv", header=None, index=False)

test_material = PowerLawMaterial(68e9, 300e6, 0.01, "baseline")
test_material.plastic_behavior()
