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

def find_localization(force):
    # Find peak force
    peak_force = np.max(force)

    # Find point of peak force
    peak_location = list(force).index(peak_force)

    return peak_location

def save_analytical_figures(dimensions):
    alphas = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    if dimensions == 3:
        for alpha in alphas:
            print(alpha)
            three_dimensional_strain(alpha, "save_all")
    elif dimensions == 2:
        for alpha in alphas:
            print(alpha)
            two_dimensional_strain(alpha, "save_all")
    else:
        raise ValueError("Input (3) or (2)-dimensional strain state assumption")

def closest_eps_finder(ana_eps, target_eps):
    eps_difference = abs(ana_eps - target_eps)
    minimum_difference = np.min(eps_difference)
    minimum_location = list(eps_difference).index(minimum_difference)

    return int(minimum_location)


def three_dimensional_strain(alpha, function):
    # Array of all possible e23 values
    small_strains = np.linspace(0, 0.01, 51)
    big_strains = np.linspace(0.01, 2.0, 400)

    epsilon_23 = np.concatenate((small_strains, big_strains[1:]))

    # Principal rotation angle
    phi = 0.5 * np.arctan(4 / alpha)

    # All other general strain components
    epsilon_11 = -epsilon_23 * (cos2(phi) - sin2(phi)) / ((1 + 2 * eta_r) * np.sin(phi) * np.cos(phi))
    epsilon_22 = -epsilon_11 * (1 + eta_r)
    epsilon_33 = epsilon_11 * eta_r

    # Principal strains
    epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi) + epsilon_33 * sin2(phi)
    epsilon_2 = epsilon_11
    epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi) + epsilon_33 * cos2(phi)

    # Equivalent principal strain and von Mises stress
    eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
    vm_stress = yield_strength * (elastic_modulus * eps / yield_strength) ** hardening_n / 10 ** 6

    # Create filepath based on alpha
    alpha_dictionary = {0.25:"025", 0.375:"0375", 0.5:"05", 0.75:"075", 1.0:"10", 1.25:"125", 1.5:"15", 2.0:"20",
                        2.5:"25", 3.0:"30"}

    filepath = rf"StressStrainCurves/Alpha{alpha_dictionary[alpha]}.csv"
    forcepath = rf"AlphaResults/Alpha{alpha_dictionary[alpha]}.csv"

    # --- Plot comparison between analytical and numerical stress-strain curves ---
    if function == "stress_comparison" or function == "save_all":

        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv(filepath, delimiter=";")

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
        plt.title(fr"Comparison between analytical and numerical results in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"ThreeDimensionalAnalytical/StressStrainAlpha{alpha_dictionary[alpha]}")
        else:
            plt.show()


    # --- Plot comparison between analytical and numerical moment curves
    if function == "moment_comparison" or function == "save_all":

        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6

        # Numerical method
        abq_data = pd.read_csv(filepath, delimiter=";")
        moment_data = pd.read_csv(forcepath, delimiter=";")

        # Extract data
        abq_moment = -(moment_data["RM2 [Nm]"].dropna()).to_numpy()
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_moment)]

        # Actual localization point (peak force)
        localization_location = find_localization(abq_moment)
        localization_x, localization_y = abq_eps[localization_location], abq_moment[localization_location]


        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, moment, label="Analytical")
        plt.plot(abq_eps, abq_moment, label="Numerical")
        plt.scatter(localization_x, localization_y, color="tab:orange", marker="x", label="Numerical localization")
        plt.xlim(-0.005, 1.1 * localization_x)
        # plt.ylim(0, 1.1 * max(np.amax(moment), np.amax(-abq_moment)))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"Resultant moment $M$ [Nm]")
        plt.title(fr"Resultant moment $M$ in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"ThreeDimensionalAnalytical/MomentComparison{alpha_dictionary[alpha]}")
        else:
            plt.show()

    # --- Plot comparison between analytical and numerical moment curves
    if function == "force_comparison" or function == "save_all":

        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6
        force = moment * outer_radius * alpha * initial_area / polar_moment

        # Numerical method
        abq_data = pd.read_csv(filepath, delimiter=";")
        force_data = pd.read_csv(forcepath, delimiter=";")

        # Extract data
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_force)]

        # Actual localization point (peak force)
        localization_location = find_localization(abq_force)
        localization_x, localization_y = abq_eps[localization_location], abq_force[localization_location]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, force, label="Analytical")
        plt.plot(abq_eps, abq_force, label="Numerical")
        plt.scatter(localization_x, localization_y, color="tab:orange", marker="x", label="Numerical localization")
        plt.xlim(-0.005, 1.1 * localization_x)
        # plt.ylim(0, 1.1 * max(np.amax(force), np.amax(abq_force)))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"Resultant force $F$ [N]")
        plt.title(fr"Resultant force $F$ in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"ThreeDimensionalAnalytical/ForceComparison{alpha_dictionary[alpha]}")
        else:
            plt.show()

    # --- Make localization prediction based on Considère criterion ---
    if function == "localization_prediction" or function == "save_all":
        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv(filepath, delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Localization points for analytical and numerical
        ana_derivative, ana_loc_x, ana_loc_y = considere_criterion(eps, vm_stress)
        num_derivative, num_loc_x, num_loc_y = considere_criterion(abq_eps, abq_vm)

        # Actual localization point (peak force)
        force_data = pd.read_csv(forcepath, delimiter=";")
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
        plt.title(fr"Localization prediction, $\alpha$ = {alpha}")
        plt.legend(loc="lower right")
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"ThreeDimensionalAnalytical/LocalizationPrediction{alpha_dictionary[alpha]}")
        else:
            plt.show()

    if function == "comparison":
        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6
        force = moment * outer_radius * alpha * initial_area / polar_moment

        return moment, force, eps

    if function == "localization":
        # There is only one localization detection methods in 3D strain states: Considere
        ana_derivative, ana_loc_x, ana_loc_y = considere_criterion(eps, vm_stress)

        return ana_loc_x

    return None


def two_dimensional_strain(alpha, function):
    # Array of all possible e23 values
    small_strains = np.linspace(0, 0.01, 101)
    big_strains = np.linspace(0.01, 2.0, 8000)

    epsilon_23 = np.concatenate((small_strains, big_strains[1:]))

    # Principal rotation angle
    phi = 0.5 * np.arctan(4 / alpha)

    # All other general strain components
    epsilon_22 = epsilon_23 * ((cos2(phi) - sin2(phi)) / (np.sin(phi) * np.cos(phi)))

    # Principal strains
    epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi)
    epsilon_2 = 0
    epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi)

    # Equivalent principal strain and von Mises stress
    eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
    vm_stress = yield_strength * (elastic_modulus * eps / yield_strength) ** hardening_n / 10 ** 6

    # Create filepath based on alpha
    alpha_dictionary = {0.25: "025", 0.375: "0375", 0.5: "05", 0.75: "075", 1.0: "10", 1.25: "125", 1.5: "15",
                        2.0: "20",
                        2.5: "25", 3.0: "30"}

    filepath = rf"StressStrainCurves/Alpha{alpha_dictionary[alpha]}.csv"
    forcepath = rf"AlphaResults/Alpha{alpha_dictionary[alpha]}.csv"

    # --- Plot comparison between analytical and numerical stress-strain curves ---
    if function == "stress_comparison" or function == "save_all":

        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv(filepath, delimiter=";")

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
        plt.title(fr"Comparison between analytical and numerical results in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"TwoDimensionalAnalytical/StressStrainAlpha{alpha_dictionary[alpha]}")
        else:
            plt.show()

    # --- Plot comparison between analytical and numerical moment curves
    if function == "moment_comparison" or function == "save_all":

        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6

        # Numerical method
        abq_data = pd.read_csv(filepath, delimiter=";")
        moment_data = pd.read_csv(forcepath, delimiter=";")

        # Extract data
        abq_moment = -(moment_data["RM2 [Nm]"].dropna()).to_numpy()
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_moment)]

        # Actual localization point (peak force)
        localization_location = find_localization(abq_moment)
        localization_x, localization_y = abq_eps[localization_location], abq_moment[localization_location]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, moment, label="Analytical")
        plt.plot(abq_eps, abq_moment, label="Numerical")
        plt.scatter(localization_x, localization_y, color="tab:orange", marker="x", label="Numerical localization")
        plt.xlim(-0.005, 1.1 * localization_x)
        # plt.ylim(0, 1.1 * max(np.amax(moment), np.amax(-abq_moment)))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"Resultant moment $M$ [Nm]")
        plt.title(fr"Resultant moment $M$ in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"TwoDimensionalAnalytical/MomentComparison{alpha_dictionary[alpha]}")
        else:
            plt.show()

    # --- Plot comparison between analytical and numerical moment curves
    if function == "force_comparison" or function == "save_all":

        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6
        force = moment * outer_radius * alpha * initial_area / polar_moment

        # Numerical method
        abq_data = pd.read_csv(filepath, delimiter=";")
        force_data = pd.read_csv(forcepath, delimiter=";")

        # Extract data
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_force)]

        # Actual localization point (peak force)
        localization_location = find_localization(abq_force)
        localization_x, localization_y = abq_eps[localization_location], abq_force[localization_location]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(eps, force, label="Analytical")
        plt.plot(abq_eps, abq_force, label="Numerical")
        plt.scatter(localization_x, localization_y, color="tab:orange", marker="x", label="Numerical localization")
        plt.xlim(-0.005, 1.1 * localization_x)
        plt.ylim(0, 1.25 * max(localization_y, force[closest_eps_finder(eps, localization_x)]))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"Resultant force $F$ [N]")
        plt.title(fr"Resultant force $F$ in tension-torsion, $\alpha$ = {alpha}")
        plt.legend()
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"TwoDimensionalAnalytical/ForceComparison{alpha_dictionary[alpha]}")
        else:
            plt.show()

    # --- Make localization prediction based on Considère criterion ---
    if function == "localization_prediction" or function == "save_all":
        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv(filepath, delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Localization points for analytical and numerical
        ana_derivative, ana_loc_x, ana_loc_y = considere_criterion(eps, vm_stress)
        num_derivative, num_loc_x, num_loc_y = considere_criterion(abq_eps, abq_vm)

        # Actual localization point (peak force)
        force_data = pd.read_csv(forcepath, delimiter=";")
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
        plt.scatter(localization_x, localization_y, label=r"Numerical localization, $peak$", color="tab:red",
                    marker="x")
        plt.xlim(-0.005, 1.5 * localization_x)
        plt.ylim(0, 1.5 * np.amax(vm_stress))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"von Mises equivalent stress $\bar{\sigma}$ [MPa]")
        plt.title(fr"Localization prediction, $\alpha$ = {alpha}")
        plt.legend(loc="lower right")
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"TwoDimensionalAnalytical/LocalizationPrediction{alpha_dictionary[alpha]}")
        else:
            plt.show()

    if function == "swift_criterion" or function == "save_all":
        # Determine some "mean" strain ratio
        strain_ratio = epsilon_3 / epsilon_1
        mean_strain_ratio = np.mean(strain_ratio[~np.isnan(strain_ratio)])

        # Then write major necking strain
        major_necking_strain = 2 * hardening_n * ((1 + strain_ratio + strain_ratio ** 2)
                                                  / (2 + strain_ratio + strain_ratio ** 2 + 2 * strain_ratio ** 3))

        print(strain_ratio, major_necking_strain)


    if function == "hill_criterion":
        # Determine some "mean" strain ratio
        strain_ratio = epsilon_3 / epsilon_1
        mean_strain_ratio = np.mean(strain_ratio[~np.isnan(strain_ratio)])

        # Then write major necking strain
        major_necking_strain = hardening_n / (1 + strain_ratio)

        print(strain_ratio, major_necking_strain)

    if function == "comparison":
        # Analytical method
        secant_modulus = vm_stress / eps
        secant_modulus[np.isnan(secant_modulus)] = 0
        sigma_1 = secant_modulus * (4 * epsilon_1 + 2 * epsilon_3) / 3
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6
        force = moment * outer_radius * alpha * initial_area / polar_moment

        return moment, force, eps

    if function == "localization":
        # There are three different localization detection methods in 2D strain states:
        # Considere, Hill, Swift
        # Numerically, there's Considere, peak load and peak moment
        # Read in .csv data as pandas DataFrame
        abq_data = pd.read_csv(filepath, delimiter=";")

        # Extract data
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

        # Localization points for analytical and numerical
        ana_derivative, ana_loc_x, ana_loc_y = considere_criterion(eps, vm_stress)
        num_derivative, num_loc_x, num_loc_y = considere_criterion(abq_eps, abq_vm)

        # Actual localization point (peak force)
        force_data = pd.read_csv(forcepath, delimiter=";")
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
        force_localization_location = find_localization(abq_force)
        force_localization = abq_eps[force_localization_location]

        abq_moment = -(force_data["RM2 [Nm]"].dropna()).to_numpy()
        moment_localization_location = find_localization(abq_moment)
        moment_localization = abq_eps[moment_localization_location]

        # Determine some "mean" strain ratio
        strain_ratio = epsilon_3 / epsilon_1
        mean_strain_ratio = np.mean(strain_ratio[~np.isnan(strain_ratio)])

        # Then Swift abd Hill localization strain
        swift_major = 2 * hardening_n * ((1 + mean_strain_ratio + mean_strain_ratio ** 2)
                                                  / (2 + mean_strain_ratio + mean_strain_ratio ** 2 + 2 * mean_strain_ratio ** 3))
        swift_result = eps[closest_eps_finder(epsilon_1, swift_major)]
        hill_major = hardening_n / (1 + mean_strain_ratio)
        hill_result = eps[closest_eps_finder(epsilon_1, hill_major)]

        return ana_loc_x, num_loc_x, force_localization, moment_localization, swift_result, hill_result

    return None


def three_two_dimensions_comparison(function=None):
    alphas = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    for alpha in alphas:
        print(alpha)

        # Create filepath based on alpha
        alpha_dictionary = {0.25: "025", 0.375: "0375", 0.5: "05", 0.75: "075", 1.0: "10", 1.25: "125", 1.5: "15",
                            2.0: "20",
                            2.5: "25", 3.0: "30"}

        filepath = rf"StressStrainCurves/Alpha{alpha_dictionary[alpha]}.csv"
        forcepath = rf"AlphaResults/Alpha{alpha_dictionary[alpha]}.csv"

        # Get analytical data
        three_moment, three_force, three_eps = three_dimensional_strain(alpha, "comparison")
        two_moment, two_force, two_eps = two_dimensional_strain(alpha, "comparison")

        # Get numerical data
        abq_data = pd.read_csv(filepath, delimiter=";")
        # Actual localization point (peak force)
        force_data = pd.read_csv(forcepath, delimiter=";")
        abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()

        # Extract data
        abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6
        abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()[:len(abq_force)]

        localization_location = find_localization(abq_force)
        localization_x, localization_y = abq_eps[localization_location], abq_force[localization_location]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(three_eps, three_force, label="Analytical, 3D strain")
        plt.plot(two_eps, two_force, label="Analytical, 2D strain")
        plt.plot(abq_eps, abq_force, "--", label="Numerical")
        plt.scatter(localization_x, localization_y, color="tab:green", label="Numerical localization")
        plt.xlim(-0.005, 1.1 * localization_x)
        plt.ylim(0, 1.25 * max(localization_y,
                               three_force[closest_eps_finder(three_eps, localization_x)],
                               two_force[closest_eps_finder(two_eps, localization_x)]))
        plt.xlabel(r"Equivalent plastic strain $\bar{\varepsilon}$ [-]")
        plt.ylabel(r"Resultant force $F$ [N]")
        plt.title(fr"Comparison of force $F$ in strain state dimensionality, $\alpha$ = {alpha}")
        plt.legend(loc="lower right")
        plt.grid()

        if function == "save_all":
            plt.savefig(fr"DimensionComparison/DimensionComparisonAlpha{alpha_dictionary[alpha]}")
        else:
            plt.show()


def localization_prediction(phi_type):
    alphas = [0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results = np.empty((len(alphas), 7))

    for n, alpha in enumerate(alphas):
        if phi_type == "approximation":
            two_results = list(two_dimensional_strain(alpha, "localization"))
        elif phi_type == "exact":
            two_results = list(two_dimensional_exact_phi(alpha, "localization"))

        three_result = [three_dimensional_strain(alpha, "localization")]

        alpha_result = np.concatenate((two_results, three_result))
        results[n] = alpha_result

    considere_2d, considere_numerical, peak_load, peak_moment, swift_2d, hill_2d, considere_3d = results.transpose()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(alphas, considere_2d, label="2D Ana., Considère")
    plt.plot(alphas, considere_numerical, label="Numerical, Considère")
    plt.plot(alphas, peak_load, "--", label="Numerical, Peak load", marker="o")
    plt.plot(alphas, peak_moment, "--", label="Numerical, Peak moment", marker="o")
    plt.plot(alphas, swift_2d, label="2D Ana., Swift")
    plt.plot(alphas, hill_2d, "--", label="2D Ana., Hill", marker="o")
    plt.plot(alphas, considere_3d, label="3D Ana., Considère")
    plt.xlabel(r"Loading ratio $\alpha$")
    plt.ylabel(r"Equivalent plastic strain $\bar{\varepsilon}$")
    plt.title("Comparison of various applied localization criteria")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=150)
    # plt.plot(alphas, (considere_2d - peak_numerical) / peak_numerical * 100, label="2D Ana., Considère")
    # plt.plot(alphas, (considere_numerical - peak_numerical) / peak_numerical * 100, label="Numerical, Considère")
    plt.plot(alphas, (peak_load - hill_2d) / hill_2d * 100, "--",
             label="Numerical, Peak load", marker="o", color="tab:green")
    plt.plot(alphas, (peak_moment - hill_2d) / hill_2d * 100, "--",
             label="Numerical, Peak moment", marker="o", color="tab:red")
    # plt.plot(alphas, (swift_2d - peak_numerical) / peak_numerical * 100, label="2D Ana., Swift")
    plt.plot(alphas, (hill_2d - hill_2d) / hill_2d * 100, "--",
             label="2D Ana., Hill", marker="o", color="tab:brown")
    # plt.plot(alphas, (considere_3d - peak_numerical) / peak_numerical * 100, label="3D Ana., Considère")
    # plt.ylim(-110, 10)
    plt.xlabel(r"Loading ratio $\alpha$ [-]")
    plt.ylabel(r"Prediction error in $\bar{\varepsilon}$ [%]")
    plt.title("Comparison of various applied localization criteria")
    plt.legend()
    plt.grid()
    plt.show()


def two_dimensional_exact_phi(alpha, function):
    # Array of all possible e23 values
    small_strains = np.linspace(0, 0.01, 101)
    big_strains = np.linspace(0.01, 2.0, 8000)

    epsilon_23 = np.concatenate((small_strains, big_strains[1:]))

    # Principal rotation angle
    phi_dictionary = {0.25:0.753648, 0.375:0.736434, 0.5:0.71911, 0.75:0.686111, 1.0:0.65193, 1.25:0.618939,
                      1.5:0.585027, 2:0.525188, 2.5:0.474167, 3:0.422358}
    phi = phi_dictionary[alpha]

    # All other general strain components
    epsilon_22 = epsilon_23 * ((cos2(phi) - sin2(phi)) / (np.sin(phi) * np.cos(phi)))

    # Principal strains
    epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi)
    epsilon_2 = 0
    epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi)

    # Equivalent principal strain and von Mises stress
    eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
    vm_stress = yield_strength * (elastic_modulus * eps / yield_strength) ** hardening_n / 10 ** 6

    # Create filepath based on alpha
    alpha_dictionary = {0.25: "025", 0.375: "0375", 0.5: "05", 0.75: "075", 1.0: "10", 1.25: "125", 1.5: "15",
                        2.0: "20",
                        2.5: "25", 3.0: "30"}

    filepath = rf"StressStrainCurves/Alpha{alpha_dictionary[alpha]}.csv"
    forcepath = rf"AlphaResults/Alpha{alpha_dictionary[alpha]}.csv"

    # There are three different localization detection methods in 2D strain states:
    # Considere, Hill, Swift
    # Also numerically Considere and peak load
    # Read in .csv data as pandas DataFrame
    abq_data = pd.read_csv(filepath, delimiter=";")

    # Extract data
    abq_eps = (abq_data["EPS [-]"].dropna()).to_numpy()
    abq_vm = (abq_data["VM [Pa]"].dropna()).to_numpy() / 10 ** 6

    # Localization points for analytical and numerical
    ana_derivative, ana_loc_x, ana_loc_y = considere_criterion(eps, vm_stress)
    num_derivative, num_loc_x, num_loc_y = considere_criterion(abq_eps, abq_vm)

    # Actual localization point (peak force)
    force_data = pd.read_csv(forcepath, delimiter=";")
    abq_force = (force_data["RF Mag [N]"].dropna()).to_numpy()
    force_localization_location = find_localization(abq_force)
    force_localization = abq_eps[force_localization_location]

    abq_moment = -(force_data["RM2 [Nm]"].dropna()).to_numpy()
    moment_localization_location = find_localization(abq_moment)
    moment_localization = abq_eps[moment_localization_location]

    # Determine some "mean" strain ratio
    strain_ratio = epsilon_3 / epsilon_1
    mean_strain_ratio = np.mean(strain_ratio[~np.isnan(strain_ratio)])

    # Then Swift abd Hill localization strain
    swift_major = 2 * hardening_n * ((1 + mean_strain_ratio + mean_strain_ratio ** 2)
                                     / (2 + mean_strain_ratio + mean_strain_ratio ** 2 + 2 * mean_strain_ratio ** 3))
    swift_result = eps[closest_eps_finder(epsilon_1, swift_major)]
    hill_major = hardening_n / (1 + mean_strain_ratio)
    hill_result = eps[closest_eps_finder(epsilon_1, hill_major)]

    return ana_loc_x, num_loc_x, force_localization, moment_localization, swift_result, hill_result









# three_two_dimensions_comparison("save_all")

# two_dimensional_strain(3.0, "hill_criterion")
# save_analytical_figures(2)
localization_prediction("exact")
plt.close('all')
