# ----- Import libraries -----
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ----- Import class definition -----
import AnalyticalTensionTorsionClean as Attc

# ----- Class definition -----
class NumericalHill(Attc.TensionTorsionSample):
    def __init__(self):
        super().__init__()

    def strain_ratio(self, alpha):
        path = fr"HillData/Alpha{self.alphas_text_dictionary[alpha]}/3Ratio.csv"
        data = pd.read_csv(path)

        peeq = (data["PEEQMAX"].values + data["PEEQMIN"].values) / 2
        major_strain = (data["PE1MAX"].values + data["PE1MIN"].values) / 2
        minor_strain = (data["PE2MAX"].values + data["PE2MIN"].values) / 2
        strain_ratio = minor_strain / major_strain

        return peeq, strain_ratio

    def non_proportionality(self):

        plt.figure(figsize=(8, 5), dpi=150)
        for alpha in self.alphas:
            peeq, strain_ratio = self.strain_ratio(alpha)
            triaxiality = (1 + strain_ratio) / (3 ** 0.5 * (1 + strain_ratio + strain_ratio ** 2) ** 0.5)
            plt.plot(triaxiality, peeq, label=fr"$\alpha$ = {alpha}")
            plt.xlabel(r"Stress triaxiality $\eta$ [-]")
            plt.ylabel(r"Equivalent plastic strain $\bar{\varepsilon}_p$ [-]")
            plt.title("Non-proportionality of loading")
            plt.grid()

        plt.legend()
        plt.grid()
        plt.show()







    def hill_prediction(self, alpha):
        peeq, strain_ratio = self.strain_ratio(alpha)
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
        plt.plot(self.alphas, force_localization, "--", label="Axisym., Peak load", marker="x")
        plt.plot(self.alphas, moment_localization, "--", label="Axisym., Peak moment", marker="x")
        plt.plot(self.alphas, shell_hill, label=f"Shells, Hill", marker="x")
        plt.xlabel(r"Loading ratio $\alpha$")
        plt.ylabel(r"PEEQ at localization $\bar{\varepsilon}_{loc}$")
        plt.title("Comparison of various applied localization criteria")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.alphas, (force_localization - hill_result) / hill_result * 100, "--",
                 label="Axisym., Peak load", marker="o", color="tab:green")
        plt.plot(self.alphas, (moment_localization - hill_result) / hill_result * 100, "--",
                 label="Axisym., Peak moment", marker="o", color="tab:red")
        plt.plot(self.alphas, (hill_result - hill_result) / hill_result * 100, "--",
                 label=f"{strain_state_type} Ana., Hill", marker="o", color="tab:brown")
        plt.plot(self.alphas, (shell_hill - hill_result) / hill_result * 100, "--",
                 label="Shells, Hill", marker="o", color="tab:pink")
        plt.xlabel(r"Loading ratio $\alpha$ [-]")
        plt.ylabel(r"Prediction error in $\bar{\varepsilon}_{loc}$ [%]")
        plt.title("Comparison of various applied localization criteria")
        plt.legend()
        plt.grid()
        plt.show()







hill = NumericalHill()
hill.non_proportionality()