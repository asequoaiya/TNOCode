# ----- Import libraries -----
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----- Constants (material, geometry) -----
elastic_modulus = 68e9
yield_strength = 267e6
hardening_n = 0.057
yield_strain = (yield_strength / elastic_modulus)

outer_radius = 0.02319
inner_radius = 0.02217
eta_r = (outer_radius - inner_radius) / (outer_radius + inner_radius)
initial_area = np.pi * (outer_radius ** 2 - inner_radius ** 2)
polar_moment = np.pi / 2 * (outer_radius ** 4 - inner_radius ** 4)


# ----- Function definitions -----
def cos2(value):
    return np.cos(value) ** 2

def sin2(value):
    return np.sin(value) ** 2

def considère_criterion(strain, stress):
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

def find_localization(force):
    # Find peak force
    peak_force = np.max(force)

    # Find point of peak force
    peak_location = list(force).index(peak_force)

    return peak_location

def save_analytical_figures():
    stress_strain_curve("save_all")


def stress_strain_curve(function):
    # Array of all possible e22 values
    small_strains = np.linspace(0, 0.01, 11)
    big_strains = np.linspace(0.01, 0.5, 200)

    epsilon_22 = np.concatenate((small_strains, big_strains[1:]))

    # Calculate force based on component terms
    fraction_term = 2 * initial_area * yield_strength / (3 ** 0.5)
    power_term = (2 * elastic_modulus / (3 ** 0.5 * yield_strength)) ** hardening_n
    epsilon_term = epsilon_22 ** hardening_n - epsilon_22 ** (hardening_n + 1)

    force = fraction_term * power_term * epsilon_term

    # Calculate EPS and von Mises
    eps = (2 / 3 ** 0.5) * epsilon_22
    sigma_22 = force / (initial_area * (1 - epsilon_22))
    vm_stress = (3 ** 0.5 / 2) * sigma_22 / (10 ** 6)

    if function == "stress_comparison" or function == "save_all":
        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv("StressStrainCurves/AlphaPS.csv", delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Make plot
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, vm_stress, label="Analytical")
        plt.plot(abq_eps, abq_vm, label="Numerical")
        plt.xlim(-0.005, np.amax(eps))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"von Mises equivalent stress $\bar{\sigma}$ [MPa]")
        plt.title(fr"Comparison between analytical and numerical results in PS tension")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(r"ThreeDimensionalAnalytical/StressStrainPS.png")
        else:
            plt.show()

    # --- Make localization prediction based on Considère criterion ---
    if function == "localization_prediction" or function == "save_all":
        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv("StressStrainCurves/AlphaPS.csv", delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Localization points for analytical and numerical
        ana_derivative, ana_loc_x, ana_loc_y = considère_criterion(eps, vm_stress)
        num_derivative, num_loc_x, num_loc_y = considère_criterion(abq_eps, abq_vm)

        # Actual localization point (peak force)
        force_data = pd.read_csv("AlphaResults/AlphaPS.csv", delimiter=";")
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
        localization_location = find_localization(abq_force)
        localization_x, localization_y = abq_eps[localization_location], abq_vm[localization_location]

        # Make plot
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, vm_stress, label="Analytical stress")
        plt.plot(eps, ana_derivative, ":", label="Analytical derivative")
        plt.scatter(ana_loc_x, ana_loc_y, label="Analytical localization, Considère", color="tab:orange")
        plt.plot(abq_eps, abq_vm, label="Numerical stress")
        plt.plot(abq_eps, num_derivative, ":", label="Numerical derivative")
        plt.scatter(num_loc_x, num_loc_y, label="Numerical localization, Considère", color="tab:red")
        plt.scatter(localization_x, localization_y, label=r"Numerical localization, $peak$", color="tab:red", marker="x")
        plt.xlim(-0.005, 1.5 * localization_x)
        plt.ylim(0, 1.5 * np.amax(vm_stress))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"von Mises equivalent stress $\bar{\sigma}$ [MPa]")
        plt.title(fr"Localization prediction, PS tension")
        plt.legend(loc="lower right")
        plt.grid()

        if function == "save_all":
            plt.savefig(r"ThreeDimensionalAnalytical/LocalizationPredictionPS")
        else:
            plt.show()





stress_strain_curve("save_all")



