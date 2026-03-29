# ----- Import libraries -----
import numpy as np
import pandas as pd
import math

# ----- Class definition -----
class NumericalHill:
    def __init__(self, path, hardening_parameter):
        self.path = path
        self.hardening_parameter = hardening_parameter

    def hill_prediction(self):
        data = pd.read_csv(self.path)

        peeq = (data["PEEQMAX"].values + data["PEEQMIN"].values) / 2
        major_strain = (data["PE1MAX"].values + data["PE1MIN"].values) / 2
        minor_strain = (data["PE2MAX"].values + data["PE2MIN"].values) / 2
        strain_ratio = minor_strain / major_strain

        peeq_increments = np.diff(peeq)

        failure_indicator = 0

        for (n, increment) in enumerate(peeq_increments):
            if not math.isnan(strain_ratio[n + 1]):
                failure_strain = self.hardening_parameter / (1 + strain_ratio[n + 1])
                failure_increment = increment / failure_strain
                failure_indicator += failure_increment

            if failure_indicator > 1:
                return peeq[n], failure_indicator

        return failure_indicator



hill = NumericalHill(r"HillData/Alpha10/3Ratio.csv", 0.057)
print(hill.hill_prediction())