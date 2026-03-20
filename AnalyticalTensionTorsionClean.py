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
    def __init__(self, strain_state_type):
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
        big_strains = np.linspace(0.01, 2.0, 400)
        epsilon_23 = np.concatenate((small_strains, big_strains[1:]))

        phi = None
        if phi_type == "approximation":
            phi = 0.5 * np.arctan(4 / alpha)
        elif phi_type == "exact":
            phi = self.phi_dictionary[alpha]

        if strain_state_type == "2D" or "2d" or 2:
            # All other general strain components
            epsilon_22 = epsilon_23 * ((cos2(phi) - sin2(phi)) / (np.sin(phi) * np.cos(phi)))

            # Principal strains
            epsilon_1 = epsilon_22 * cos2(phi) + 2 * epsilon_23 * np.sin(phi) * np.cos(phi)
            epsilon_2 = 0
            epsilon_3 = epsilon_22 * sin2(phi) - 2 * epsilon_23 * np.sin(phi) * np.cos(phi)

            # Equivalent plastic strain and von Mises stress
            eps = (2 / 3) ** 0.5 * (epsilon_1 ** 2 + epsilon_2 ** 2 + epsilon_3 ** 2) ** 0.5
            vm_stress = yield_strength * (elastic_modulus * eps / yield_strength) ** hardening_n / 10 ** 6

            return eps, vm_stress, epsilon_1, epsilon_2, epsilon_3

        elif strain_state_type == "3D" or "3d" or 3:
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
        sigma_3 = -2 * secant_modulus * epsilon_2 - sigma_1

        # Solve s23
        sigma_23 = ((sigma_1 - sigma_3) ** 2 / (4 * (1 + alpha ** 2 / 16))) ** 0.5

        # Solve moment and force
        geometry_term = (2 * np.pi / (hardening_n + 3)
                         * (outer_radius ** (hardening_n + 3) - inner_radius ** (hardening_n + 3)))
        moment = (sigma_23 / outer_radius ** hardening_n) * geometry_term * 10 ** 6
        force = moment * outer_radius * alpha * initial_area / polar_moment

        return moment, force

    # Stress-strain, force, and moment curves






