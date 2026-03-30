# ----- Import libraries -----
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ----- Import class definition -----
import AnalyticalTensionTorsionClean as attc

# ----- Class definition -----
class NumericalHill(attc.TensionTorsionSample):
    def __init__(self):
        super().__init__()

    def hill_prediction(self, alpha):
        path = fr"HillData/Alpha{self.alphas_text_dictionary[alpha]}/3Ratio.csv"
        data = pd.read_csv(path)

        peeq = (data["PEEQMAX"].values + data["PEEQMIN"].values) / 2
        major_strain = (data["PE1MAX"].values + data["PE1MIN"].values) / 2
        minor_strain = (data["PE2MAX"].values + data["PE2MIN"].values) / 2
        strain_ratio = minor_strain / major_strain

        peeq_increments = np.diff(peeq)

        failure_indicator = 0

        for (n, increment) in enumerate(peeq_increments):
            if not math.isnan(strain_ratio[n + 1]):
                failure_strain = self.hardening_n / (1 + strain_ratio[n + 1])
                failure_increment = increment / failure_strain
                failure_indicator += failure_increment

            if failure_indicator > 1:
                return peeq[n], failure_indicator

        return 0, failure_indicator

    def hill_comparison(self, phi_type, strain_state_type):
        results = np.empty((len(self.alphas), 7))

        for (n, alpha) in enumerate(self.alphas):
            (considere_analytical, swift_result,
             hill_result, considere_numerical,
             force_localization, moment_localization) = self.localization_prediction(alpha, phi_type, strain_state_type)

            shell_hill = self.hill_prediction(alpha)[0]

            result_vector = (considere_analytical, swift_result, hill_result, considere_numerical,
                             force_localization, moment_localization, shell_hill)

            results[n] = result_vector

        (considere_analytical, swift_result, hill_result,
         considere_numerical, force_localization, moment_localization, shell_hill) = results.transpose()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.alphas, considere_analytical, label=f"{strain_state_type} Ana., Considère")
        plt.plot(self.alphas, swift_result, label=f"{strain_state_type} Ana., Swift")
        plt.plot(self.alphas, hill_result, "--", label=f"{strain_state_type} Ana., Hill", marker="o")
        plt.plot(self.alphas, considere_numerical, label="Axisym., Considère")
        plt.plot(self.alphas, force_localization, "--", label="Axisym., Peak load", marker="o")
        plt.plot(self.alphas, moment_localization, "--", label="Axisym., Peak moment", marker="o")
        plt.plot(self.alphas, shell_hill, label=f"Shells, Hill", marker="o")
        plt.xlabel(r"Loading ratio $\alpha$")
        plt.ylabel(r"PEEQ at localization $\bar{\varepsilon}_{loc}$")
        plt.title("Comparison of various applied localization criteria")
        plt.legend()
        plt.grid()
        plt.show()







hill = NumericalHill()
hill.hill_comparison("approximation", "3D")