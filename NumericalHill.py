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
        self.shell_sizes = [0.1, 0.3, 1.0, 3.0]
        self.shell_size_dictionary = {0.1:"01", 0.3: "03", 1.0: "1", 3.0:"3"}

    @staticmethod
    def triaxiality(ratio):
        return (1 + ratio) / (3 ** 0.5 * (1 + ratio + ratio ** 2) ** 0.5)

    def strain_ratio(self, alpha, shell_size):
        path = fr"HillData/Alpha{self.alphas_text_dictionary[alpha]}/{self.shell_size_dictionary[shell_size]}Ratio.csv"
        data = pd.read_csv(path)

        peeq = (data["PEEQMAX"].values + data["PEEQMIN"].values) / 2
        major_strain = (data["PE1MAX"].values + data["PE1MIN"].values) / 2
        minor_strain = (data["PE2MAX"].values + data["PE2MIN"].values) / 2
        strain_ratio = minor_strain / major_strain

        return peeq, strain_ratio

    def non_proportionality(self, shell_size):
        # Initialize figure
        plt.figure(figsize=(8, 5), dpi=150)

        # Plot loading path for each value of alpha
        for alpha in self.alphas:
            peeq, strain_ratio = self.strain_ratio(alpha, shell_size)
            triaxiality = self.triaxiality(strain_ratio)

            plt.plot(triaxiality, peeq, label=fr"$\alpha$ = {alpha}")

            # Plot localization points
            localization_peeq, localization_indicator, localization_strain_ratio = self.hill_prediction(alpha)
            localization_triaxiality = self.triaxiality(localization_strain_ratio)
            plt.scatter(localization_triaxiality, localization_peeq, marker="o")

        # Plot Hill prediction path
        hill_alphas = np.linspace(-1, 0, 1001)
        hill_triaxialities = self.triaxiality(hill_alphas)
        hill_peeq = self.hardening_n / (1 + hill_alphas)

        plt.plot(hill_triaxialities, hill_peeq, "--", label="Hill prediction", color="black")
        plt.xlabel(r"Stress triaxiality $\eta$ [-]")
        plt.ylabel(r"Equivalent plastic strain $\bar{\varepsilon}_p$ [-]")
        plt.ylim(-0.05, 1.05)
        plt.title("Non-proportionality of loading")
        plt.legend(ncol=2)
        plt.grid()
        plt.show()

    def hill_prediction(self, alpha, shell_size):
        peeq, strain_ratio = self.strain_ratio(alpha, shell_size)
        peeq_increments = np.diff(peeq)

        failure_indicator = 0

        for (n, increment) in enumerate(peeq_increments):
            if not math.isnan(strain_ratio[n + 1]):
                failure_strain = self.hardening_n / (1 + strain_ratio[n + 1])
                failure_increment = increment / failure_strain
                failure_indicator += failure_increment

            if failure_indicator > 1:
                return peeq[n], failure_indicator, strain_ratio[n + 1]

        return 0, failure_indicator

    def hill_comparison(self, phi_type, strain_state_type, shell_size):
        results = np.empty((len(self.alphas), 7))

        for (n, alpha) in enumerate(self.alphas):
            (considere_analytical, swift_result,
             hill_result, considere_numerical,
             force_localization, moment_localization) = self.localization_prediction(alpha, phi_type, strain_state_type)

            shell_hill = self.hill_prediction(alpha, shell_size)[0]

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

    def shell_convergence(self):
        # Result list
        shell_results = []

        # Get shell results for all sizes
        for size in self.shell_sizes:
            shell_results.append(self.hill_prediction(1.0, size)[0])

        # Get axisymmetric results (truth)
        (considere_analytical, swift_result,
         hill_result, considere_numerical,
         force_localization, moment_localization) = self.localization_prediction(1.0, "approximation", "3D")

        # Plot convergence of shell model and axisymmetric model results
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.shell_sizes, shell_results, marker="o", label="Shell results")
        plt.scatter(0.05, force_localization, marker="o", label="Axisym. result", color="tab:orange")
        plt.plot([-1, 10], [hill_result, hill_result], "--", label="3D analytical result", color="tab:green")
        plt.ylim(0, 1.15 * np.max(shell_results))
        plt.ylabel(r"PEEQ at localization $\bar{\varepsilon}_{loc}$ [-]")
        plt.xlabel("Element length over thickness ratio $l_e/t_e$ [-]")
        plt.xlim(-0.2, 3.2)
        plt.title(r"Convergence of shell element results from ABAQUS for $\alpha$ = 1.0")
        plt.legend()
        plt.grid()
        plt.show()


hill = NumericalHill()
hill.shell_convergence()