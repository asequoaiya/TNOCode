# ----- Import libraries -----
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ----- Import class definition -----
import AnalyticalTensionTorsionClean as Attc
plt.rcParams['font.family'] = 'arial'


# ----- Class definition -----
class NumericalHill(Attc.TensionTorsionSample):
    def __init__(self):
        super().__init__()
        self.shell_sizes = [0.1, 0.3, 1.0, 3.0, 5.0]
        self.shell_size_dictionary = {0.1:"01", 0.3: "03", 1.0: "1", 3.0:"3", 5.0:"5"}

        # Element size ratio
        self.element_numbers = np.array([100, 34, 10, 4, 2])
        self.parallel_thickness = 0.001016
        self.parallel_height = 0.01016
        self.element_height = self.parallel_height / self.element_numbers
        self.thickness_ratio = self.element_height / self.parallel_thickness

    @staticmethod
    def triaxiality(ratio):
        return (1 + ratio) / (3 ** 0.5 * (1 + ratio + ratio ** 2) ** 0.5)

    def strain_ratio(self, alpha, shell_size):
        # Read data from correct .csv file, but only if it exists
        path = fr"HillData/Alpha{self.alphas_text_dictionary[alpha]}/{self.shell_size_dictionary[shell_size]}Ratio.csv"

        if os.path.isfile(path):
            data = pd.read_csv(path, sep=None)
        else:
            return None, None

        # Average values from bottom and top integration points
        peeq = (data["PEEQMAX"].values + data["PEEQMIN"].values) / 2
        major_strain = (data["PE1MAX"].values + data["PE1MIN"].values) / 2
        minor_strain = (data["PE2MAX"].values + data["PE2MIN"].values) / 2
        strain_ratio = minor_strain / major_strain

        return peeq, strain_ratio

    def non_proportionality(self, shell_size):
        # Initialize figure
        plt.figure(figsize=(8, 5), dpi=150)

        # Make array of standard matplotlib colors as strings
        colors = np.array(list(mcolors.TABLEAU_COLORS.items()))[:, 0]

        # Plot loading path for each value of alpha
        for (n, alpha) in enumerate(self.alphas):
            # Make sure colors are the same for the scatters
            color = colors[n]

            # Get PEEQ and calculate stress triaxiality
            peeq, strain_ratio = self.strain_ratio(alpha, shell_size)

            # Check if PEEQ or strain ratio is None
            if peeq is None or strain_ratio is None:
                pass
            else:
                triaxiality = self.triaxiality(strain_ratio)
                plt.plot(triaxiality, peeq, label=fr"$\alpha$ = {alpha}", color=color)

                # Plot first nonzero PEEQ and non-nan triaxiality as start point
                nonzero_peeq = peeq[peeq != 0]
                nonnan_strain_ratio = strain_ratio[~np.isnan(strain_ratio)]
                plt.scatter(self.triaxiality(nonnan_strain_ratio[0]), nonzero_peeq[0], marker="x", color=color)

                # Plot localization points
                localization_peeq, localization_indicator, localization_strain_ratio = self.hill_prediction(alpha, shell_size)
                localization_triaxiality = self.triaxiality(localization_strain_ratio)
                plt.scatter(localization_triaxiality, localization_peeq, marker="o", color=color)

        # Plot Hill prediction path
        hill_alphas = np.linspace(-1, 0, 1001)
        hill_triaxialities = self.triaxiality(hill_alphas)
        hill_peeq = self.hardening_n / (1 + hill_alphas)
        plt.plot(hill_triaxialities, hill_peeq, "--", label="Hill prediction", color="black")

        # Make plot
        plt.xlabel(r"Stress triaxiality $\eta$ [-]")
        plt.ylabel(r"Equivalent plastic strain $\bar{\varepsilon}_p$ [-]")
        plt.ylim(-0.05, 1.05)
        plt.title(f"Non-proportionality of loading, length/thickness ratio = {self.thickness_ratio[self.shell_sizes.index(shell_size)]:.3f}")
        plt.legend(ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.show()

    #
    def hill_prediction(self, alpha, shell_size):
        # Get PEEQ and strain ratio
        peeq, strain_ratio = self.strain_ratio(alpha, shell_size)

        # Check if peeq and strain ratio exist (in case the file doesn't exist)
        if peeq is None or strain_ratio is None:
            return None, None

        # Calculate PEEQ increments for cumulative use
        peeq_increments = np.diff(peeq)

        # Set "failure" indicator to zero
        failure_indicator = 0

        # Loop through each PEEQ increment
        for (n, increment) in enumerate(peeq_increments):
            # Filter all nan values from strain ratio array
            if not math.isnan(strain_ratio[n + 1]):
                # Calculate failure strain at this strain ratio using Hill's criterion
                failure_strain = self.hardening_n / (1 + strain_ratio[n + 1])

                # Increment failure indicator using this PEEQ increment
                failure_increment = increment / failure_strain
                failure_indicator += failure_increment

            # "Failure" (localization) occurs when the indicator becomes unity
            if failure_indicator >= 1:
                # Then return PEEQ, indicator and the strain ratio
                return peeq[n], failure_indicator, strain_ratio[n + 1]

        return 0, failure_indicator

    def hill_comparison(self, phi_type, strain_state_type, shell_size):
        # Result array
        results = np.empty((len(self.alphas), 7))

        # Collect localization results
        for (n, alpha) in enumerate(self.alphas):
            # All results BUT shells
            (considere_analytical, swift_result,
             hill_result, considere_numerical,
             force_localization, moment_localization) = self.localization_prediction(alpha, phi_type, strain_state_type)

            # Shell results
            shell_hill = self.hill_prediction(alpha, shell_size)[0]

            # Save in result array
            result_vector = (considere_analytical, swift_result, hill_result, considere_numerical,
                             force_localization, moment_localization, shell_hill)
            results[n] = result_vector

        (considere_analytical, swift_result, hill_result,
         considere_numerical, force_localization, moment_localization, shell_hill) = results.transpose()

        # Make plot showing comparison
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
        plt.tight_layout()
        plt.show()

        # Make plot showing error
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
        plt.tight_layout()
        plt.show()

    def shell_convergence(self, alpha):
        # Result list
        shell_results = []
        actual_sizes = []

        # Get shell results for all sizes
        for n, size in enumerate(self.shell_sizes):
            result = self.hill_prediction(alpha, size)[0]
            if result is None:
                pass
            else:
                shell_results.append(result)
                actual_sizes.append(n)

        # Get axisymmetric results (truth)
        (considere_analytical, swift_result,
         hill_result, considere_numerical,
         force_localization, moment_localization) = self.localization_prediction(alpha, "approximation", "3D")

        # Plot convergence of shell model and axisymmetric model results
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.thickness_ratio[actual_sizes], shell_results, marker="o", label="Shell results")
        plt.plot([-1, 10], [hill_result, hill_result], "--", label="3D analytical result", color="black")
        plt.scatter(0.05, force_localization, marker="o", label="Axisym. result", color="tab:red")
        plt.ylim(0, 1.15 * max(force_localization, hill_result, np.max(shell_results)))
        plt.ylabel(r"PEEQ at localization $\bar{\varepsilon}_{loc}$ [-]")
        plt.xlabel("Element length over thickness ratio $l_e/t_e$ [-]")
        plt.xlim(-0.2, 5.2)
        plt.title(fr"Convergence of shell element results from ABAQUS for $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def shell_thickness_plot(self):
        # Shell thicknesses
        shell_thickness_data = pd.read_csv("ShellThicknesses.csv", sep=None).to_numpy().transpose()

        plt.figure(figsize=(12, 5), dpi=150)
        for (n, size) in enumerate(self.shell_sizes):
            length = shell_thickness_data[2 * n]
            thickness = shell_thickness_data[2 * n + 1] * 1000

            normalized_length = length / np.amax(length[~np.isnan(length)])

            if size == self.shell_sizes[-1]:
                plt.plot(normalized_length, thickness,
                         "--", label=f"Ratio = {self.thickness_ratio[n]:.3f}; {self.element_numbers[n]} elements")
            else:
                plt.plot(normalized_length, thickness,
                         label=f"Ratio = {self.thickness_ratio[n]:.3f}; {self.element_numbers[n]} elements")

        # Ideal thickness distribution
        ideal_thickness_data = pd.read_csv("VariableThickness.csv", sep=None, header=None).to_numpy().transpose()
        plt.plot(ideal_thickness_data[2] / np.amax(ideal_thickness_data[2]), ideal_thickness_data[3] * 1000,
                 ":", label="Ideal", color="black")

        # Make plot
        plt.grid()
        plt.ylim(0, 1100 * np.amax(shell_thickness_data[1]))
        plt.xlabel("Normalized height along the specimen [-]")
        plt.ylabel("Shell thickness [mm]")
        plt.title("Shell thickness comparison; length/thickness ratio and #elements across parallel gage section")
        plt.legend()
        plt.tight_layout()
        plt.show()


hill = NumericalHill()
# hill.shell_convergence(3.0)
hill.non_proportionality(0.3)
# hill.shell_thickness_plot()