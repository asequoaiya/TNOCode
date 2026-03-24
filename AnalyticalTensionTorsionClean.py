# ----- Import libraries -----
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ----- Static function definitions -----
def cos2(value):
    return np.cos(value) ** 2

def sin2(value):
    return np.sin(value) ** 2

def considere_criterion(strain, stress):
    # Take numerical derivative
    derivative = np.gradient(stress, strain)

    # Filter possible nan or inf when strain is zero
    derivative[np.isnan(derivative)] = 10 ** 9
    derivative[np.isinf(derivative)] = 10 ** 9

    # Find localization point
    absolute_difference = abs(stress - derivative)
    minimum = np.min(absolute_difference)
    minimum_location = list(absolute_difference).index(minimum)
    localization_x = float(strain[minimum_location])
    localization_y = float(stress[minimum_location])

    return derivative, localization_x, localization_y

def find_peak(array):
    # Find peak value
    peak_value = np.max(array)

    # Find point of peak force
    peak_location = list(array).index(peak_value)

    return peak_location

def closest_value_finder(array, target_value):
    difference = abs(array - target_value)
    minimum_difference = np.min(difference)
    minimum_location = list(difference).index(minimum_difference)

    return minimum_location


# ----- Class definition for tension-torsion sample -----
class TensionTorsionSample:
    def __init__(self):
        # Material parameters
        self.elastic_modulus = 68e9
        self.yield_strength = 267e6
        self.hardening_n = 0.057
        self.yield_strain = self.yield_strength / self.elastic_modulus

        # Geometry parameters
        self.outer_radius = 0.02319
        self.inner_radius = 0.02217
        self.eta_r = (self.outer_radius - self.inner_radius) / (self.outer_radius + self.inner_radius)
        self.initial_area = np.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)
        self.polar_moment = np.pi / 2 * (self.outer_radius ** 4 - self.inner_radius ** 4)

        # Loading ratio alpha
        self.alphas = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        self.alphas_text_dictionary = {0.25:"025", 0.375:"0375", 0.5:"05", 0.75:"075", 1.0:"10",
                                       1.25:"125", 1.5:"15", 2.0:"20", 2.5:"25", 3.0:"30"}

        # "Exact" values of phi
        self.phi_dictionary = {0.25:0.753648, 0.375:0.736434, 0.5:0.71911, 0.75:0.686111, 1.0:0.65193,
                               1.25:0.618939, 1.5:0.585027, 2:0.525188, 2.5:0.474167, 3:0.422358}

    # Solve strains
    def solve_strain_state(self, alpha, phi_type, strain_state_type):
        # Array of all possible e23 values
        small_strains = np.linspace(0, 0.01, 51)
        big_strains = np.linspace(0.01, 2.0, 2000)
        epsilon_23 = np.concatenate((small_strains, big_strains[1:]))

        phi = None
        if phi_type == "approximation":
            phi = 0.5 * np.arctan(4 / alpha)
        elif phi_type == "exact":
            phi = self.phi_dictionary[alpha]

        if strain_state_type == "2D" or strain_state_type == "2d" or strain_state_type == 2:
            # All other general strain components
            epsilon_22 = epsilon_23 * ((cos2(phi) - sin2(phi)) / (np.sin(phi) * np.cos(phi)))

            # Principal strains
            epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi)
            epsilon_2 = np.zeros(len(epsilon_22))
            epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi)

            # Equivalent plastic strain and von Mises stress
            eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
            vm_stress = self.yield_strength * (self.elastic_modulus * eps / self.yield_strength) ** self.hardening_n / 10 ** 6

            return eps, vm_stress, epsilon_1, epsilon_2, epsilon_3

        elif strain_state_type == "3D" or strain_state_type == "3d" or strain_state_type == 3:
            # All other general strain components
            epsilon_11 = -epsilon_23 * (cos2(phi) - sin2(phi)) / ((1 + 2 * self.eta_r) * np.sin(phi) * np.cos(phi))
            epsilon_22 = -epsilon_11 * (1 + self.eta_r)
            epsilon_33 = epsilon_11 * self.eta_r

            # Principal strains
            epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi) + epsilon_33 * sin2(phi)
            epsilon_2 = epsilon_11
            epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi) + epsilon_33 * cos2(phi)

            # Equivalent principal strain and von Mises stress
            eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
            vm_stress = self.yield_strength * (self.elastic_modulus * eps / self.yield_strength) ** self.hardening_n / 10 ** 6

            return eps, vm_stress, epsilon_1, epsilon_2, epsilon_3

        else:
            raise ValueError("Please select either 2D strain state (2D, 2d, 2) or 3D strain state (3D, 3d, 3)!")

    # Solve for moment and force
    def solve_moment_force(self, alpha, phi_type, strain_state_type):
        # Get strains from strain state solver
        eps, vm_stress, epsilon_1, epsilon_2, epsilon_3 = self.solve_strain_state(alpha, phi_type, strain_state_type)

        # Solve principal stresses
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_2 = np.zeros(np.shape(sigma_1))
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        # Solve s23
        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        # Solve moment and force
        geometry_term = (2 * np.pi / (self.hardening_n + 3)
                         * (self.outer_radius ** (self.hardening_n + 3) - self.inner_radius ** (self.hardening_n + 3)))
        moment = (sigma_23 / self.outer_radius ** self.hardening_n) * geometry_term * 10 ** 6
        force = moment * self.outer_radius * alpha * self.initial_area / self.polar_moment

        return moment, force, eps, sigma_1, sigma_2, sigma_3

    # Force and moment curves
    def create_figures(self, phi_type, strain_state_type, action="show"):
        # Create string to determine where to save figure (based on strain state type)
        if strain_state_type == "2D":
            folder = "TwoDimensionalAnalytical"
        elif strain_state_type == "3D":
            folder = "ThreeDimensionalAnalytical"
        else:
            folder = ""

        # Creates moment-force figures, lops for every possible value of alpha
        for alpha in self.alphas:
            moment, force, eps, sigma_1, sigma_2, sigma_3 = self.solve_moment_force(alpha, phi_type, strain_state_type)

            # Retrieve data from .csv files
            filepath = rf"StressStrainCurves/Alpha{self.alphas_text_dictionary[alpha]}.csv"
            forcepath = rf"AlphaResults/Alpha{self.alphas_text_dictionary[alpha]}.csv"
            abq_data = pd.read_csv(filepath, delimiter=";")
            load_data = pd.read_csv(forcepath, delimiter=";")

            # Extract data
            abq_moment = -(load_data["RM2 [Nm]"].dropna()).to_numpy()
            abq_force = (load_data["RF Mag [N]"].dropna()).to_numpy()
            abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_moment)]

            # Actual localization point (peak moment)
            moment_location = find_peak(abq_moment)
            moment_x, moment_y = abq_eps[moment_location], abq_moment[moment_location]

            plt.figure(figsize=(8, 5), dpi=150)
            plt.plot(eps, moment, label="Analytical")
            plt.plot(abq_eps, abq_moment, label="Numerical")
            plt.scatter(moment_x, moment_y, color="tab:orange", marker="x", label="Num. localization, moment")
            plt.xlim(-0.005, 1.1 * moment_x)
            plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
            plt.ylabel(r"Resultant moment $M$ [Nm]")
            plt.title(fr"Resultant moment $M$ in tension-torsion, $\alpha$ = {alpha}")
            plt.legend()
            plt.grid()

            if action == "save":
                plt.savefig(fr"{folder}/MomentComparison{self.alphas_text_dictionary[alpha]}")
            else:
                plt.show()

            # Actual localization point (peak force)
            force_location = find_peak(abq_force)
            force_x, force_y = abq_eps[force_location], abq_force[force_location]

            plt.figure(figsize=(8, 5), dpi=150)
            plt.plot(eps, force, label="Analytical")
            plt.plot(abq_eps, abq_force, label="Numerical")
            plt.scatter(force_x, force_y, color="tab:orange", marker="x", label="Num. localization, force")
            plt.xlim(-0.005, 1.1 * force_x)
            plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
            plt.ylabel(r"Resultant force $F$ [N]")
            plt.title(fr"Resultant force $F$ in tension-torsion, $\alpha$ = {alpha}")
            plt.legend()
            plt.grid()

            if action == "save":
                plt.savefig(fr"{folder}/ForceComparison{self.alphas_text_dictionary[alpha]}")
            else:
                plt.show()


    def localization_prediction(self, alpha, phi_type, strain_state_type):
        # There are three different localization detection methods for the analytical solution:
        # Considere, Hill, and Swift
        # For the numerical solution, there are also three:
        # Considere, peak load, and peak moment

        # --- Analytical section ---
        eps, vm_stress, epsilon_1, epsilon_2, epsilon_3 = self.solve_strain_state(alpha, phi_type, strain_state_type)

        # Considere, analytical
        ana_derivative, considere_analytical, ana_loc_y = considere_criterion(eps, vm_stress)

        # Determine some "mean" strain ratio
        strain_ratio = epsilon_3 / epsilon_1
        mean_strain_ratio = np.mean(strain_ratio[~np.isnan(strain_ratio)])

        phi = None
        if phi_type == "approximation":
            phi = 0.5 * np.arctan(4 / alpha)
        elif phi_type == "exact":
            phi = self.phi_dictionary[alpha]

        alpha_s = -(np.tan(phi) ** 2)

        # Then Swift and Hill localization strain
        swift_major = 2 * self.hardening_n * ((1 + mean_strain_ratio + mean_strain_ratio ** 2)
                                         / (2 + mean_strain_ratio + mean_strain_ratio ** 2 + 2 * mean_strain_ratio ** 3))
        swift_result = eps[closest_value_finder(epsilon_1, swift_major)]
        hill_major = self.hardening_n / (1 + mean_strain_ratio)
        hill_result = eps[closest_value_finder(epsilon_1, hill_major)]

        # --- WIP --- Different formulation for Hill ---
        moment, force, eps, sigma_1, sigma_2, sigma_3 = self.solve_moment_force(alpha, phi_type, strain_state_type)
        dsigma_1 = np.diff(sigma_1)
        depsilon_1 = np.diff(epsilon_1)

        new_hill_criterion = abs(dsigma_1 / depsilon_1 - (1 + mean_strain_ratio) * sigma_1[1:])
        # print(eps[list(new_hill_criterion).index(np.min(new_hill_criterion))])

        # --- WIP --- Stress triaxiality ---
        mean_stress = (sigma_1 + sigma_2 + sigma_3) / 3
        stress_triaxiality = mean_stress / vm_stress
        print(sigma_1, sigma_3, stress_triaxiality, alpha)


        # --- Numerical section ---
        # Read in .csv data as pandas DataFrame
        filepath = rf"StressStrainCurves/Alpha{self.alphas_text_dictionary[alpha]}.csv"
        forcepath = rf"AlphaResults/Alpha{self.alphas_text_dictionary[alpha]}.csv"
        abq_data = pd.read_csv(filepath, delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Considere, numerical
        num_derivative, considere_numerical, num_loc_y = considere_criterion(abq_eps, abq_vm)

        # Actual localization point (peak force)
        force_data = pd.read_csv(forcepath, delimiter=";")
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
        force_localization_location = find_peak(abq_force)
        force_localization = abq_eps[force_localization_location]

        abq_moment = -(force_data["RM2 [Nm]"].dropna()).to_numpy()
        moment_localization_location = find_peak(abq_moment)
        moment_localization = abq_eps[moment_localization_location]

        return (considere_analytical, swift_result, hill_result,
                considere_numerical, force_localization, moment_localization)

    def localization_figures(self, phi_type, strain_state_type):
        results = np.empty((len(self.alphas), 6))

        for n, alpha in enumerate(self.alphas):
            results[n] = self.localization_prediction(alpha, phi_type, strain_state_type)

        (considere_analytical, swift_result, hill_result,
         considere_numerical, force_localization, moment_localization) = results.transpose()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.alphas, considere_analytical, label=f"{strain_state_type} Ana., Considère")
        plt.plot(self.alphas, swift_result, label=f"{strain_state_type} Ana., Swift")
        plt.plot(self.alphas, hill_result, "--", label=f"{strain_state_type} Ana., Hill", marker="o")
        plt.plot(self.alphas, considere_numerical, label="Numerical, Considère")
        plt.plot(self.alphas, force_localization, "--", label="Numerical, Peak load", marker="o")
        plt.plot(self.alphas, moment_localization, "--", label="Numerical, Peak moment", marker="o")
        plt.xlabel(r"Loading ratio $\alpha$")
        plt.ylabel(r"Equivalent plastic strain $\bar{\varepsilon}$")
        plt.title("Comparison of various applied localization criteria")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(self.alphas, (force_localization - hill_result) / hill_result * 100, "--",
                 label="Numerical, Peak load", marker="o", color="tab:green")
        plt.plot(self.alphas, (moment_localization - hill_result) / hill_result * 100, "--",
                 label="Numerical, Peak moment", marker="o", color="tab:red")
        plt.plot(self.alphas, (hill_result - hill_result) / hill_result * 100, "--",
                 label=f"{strain_state_type} Ana., Hill", marker="o", color="tab:brown")
        plt.xlabel(r"Loading ratio $\alpha$ [-]")
        plt.ylabel(r"Prediction error in $\bar{\varepsilon}$ [%]")
        plt.title("Comparison of various applied localization criteria")
        plt.legend()
        plt.grid()
        plt.show()



sample = TensionTorsionSample()
sample.localization_figures("approximation", "3D")
plt.close("all")


