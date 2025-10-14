# ----- Import libraries -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from icecream import ic


# Function definition for cotangent
def cot(argument):
    return 1 / np.tan(argument)


# Shear band material class definition
class ShearBandMaterial:
    def __init__(self, elastic_modulus, hardening_modulus, softening_modulus, yield_strength,
                 ultimate_tensile_strength, intrinsic_length, strain_at_uts):
        self.elastic_modulus = elastic_modulus
        self.hardening_modulus = hardening_modulus
        self.softening_modulus = softening_modulus

        self.yield_strength = yield_strength
        self.ultimate_tensile_strength = ultimate_tensile_strength

        self.intrinsic_length = intrinsic_length
        self.strain_at_uts = strain_at_uts

    def elastic_shear_modulus(self):
        return self.elastic_modulus / 3

    def hardening_shear_modulus(self):
        return self.hardening_modulus / 3

    def softening_shear_modulus(self):
        return self.softening_modulus / 3


# Shear band class definition
class ShearBand:
    def __init__(self, band_type, material: ShearBandMaterial):
        self.band_type = band_type
        self.material = material

    def band_thickness_shear(self):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()
        g_s = self.material.softening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        l_cs = self.material.intrinsic_length

        numerator = -2 * g * g_t * sigma_u
        denominator = (g_s * g_t * sigma_y) + (g * g_s * (sigma_u - sigma_y))
        root = (numerator / denominator) ** 0.5
        brackets = np.pi - np.arctan(-g_s / g)

        thickness = l_cs / 2 * root * brackets

        return 2 * thickness

    def a_1(self, theta):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        top_left_bracket = (3 * (np.power(np.cos(theta), 4) + np.power(np.sin(2 * theta), 2))
                            * (sigma_u - sigma_y)
                            * (g - g_t))
        top_right_bracket = 4 * sigma_u * g_t
        numerator = 2 * sigma_u * g_t * g * (top_left_bracket + top_right_bracket)

        bottom_left_bracket = g_t * sigma_y + g * (sigma_u - sigma_y)
        bottom_right_bracket = (3 * np.power(np.cos(theta), 4)
                                * (sigma_u - sigma_y)
                                * (g - g_t)
                                + 4 * sigma_u * g_t)
        denominator = bottom_left_bracket * bottom_right_bracket

        value = numerator / denominator

        return value

    def a_2(self, theta):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()
        g_s = self.material.softening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        factor = 2 * sigma_u * g_t * g
        top_left_bracket = 2 * sigma_y * g_s * ((g_t - g) * (np.power(np.sin(2 * theta), 2)
                                                             + np.power(np.cos(theta), 4)))
        top_right_bracket = g * sigma_u * (3 * (np.power(np.sin(2 * theta), 2)
                                                + np.power(np.cos(theta), 4))
                                           * (g_s - g_t)
                                           + 4 * g_t)
        numerator = factor * (top_left_bracket + top_right_bracket)

        bottom_left_bracket = g_t * sigma_y + g * (sigma_u - sigma_y)
        bottom_right_bracket = (3 * (g_t - g) * g_s * sigma_y * (np.power(np.cos(theta), 4))
                                + 4 * g * g_t * sigma_u
                                + 3 * g * sigma_u * np.power(np.cos(theta), 4) * (g_s - g_t))
        denominator = bottom_left_bracket * bottom_right_bracket

        value = numerator / denominator

        return value

    def b_2(self, theta):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()
        g_s = self.material.softening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        factor = g_t * sigma_u
        top_left_bracket = g * sigma_u * (-64 * g * np.sin(2 * theta)
                                          + (9 * np.sin(6 * theta)
                                             + 13 * np.sin(2 * theta)
                                             - 12 * np.sin(4 * theta) * (g_s - g_t)))
        top_right_bracket = (-9 * g_s * np.sin(6 * theta)
                             + (64 * g - 13 * g_s) * np.sin(2 * theta)
                             + 12 * g_s * np.sin(4 * theta)) * (g - g_t) * sigma_y
        numerator = factor * (top_left_bracket + top_right_bracket)

        bottom_left_bracket = -32 * g_t * sigma_y + g * (sigma_u - sigma_y)
        bottom_right_bracket = (3 * (g_t - g) * g_s * sigma_y * (np.power(np.cos(theta), 4))
                                + 4 * g * g_t * sigma_u
                                + 3 * g * sigma_u * np.power(np.cos(theta), 4) * (g_s - g_t))
        denominator = bottom_left_bracket * bottom_right_bracket

        value = numerator / denominator

        return value

    def a_3(self, theta):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        top_left_bracket = (3 * np.power(np.cos(theta), 4)
                            * (g_t - g)
                            * (sigma_y - sigma_u)
                            + 4 * g_t * sigma_u)
        top_right_bracket = g_t * sigma_y + g * (sigma_u - sigma_y)
        numerator = top_left_bracket * top_right_bracket

        factor = g * sigma_u * g_t
        bottom = (3 * (3 * np.power(np.cos(theta), 4) - 4 * np.power(np.cos(theta), 2))
                  * (g_t - g)
                  * (sigma_y - sigma_u)
                  - 4 * g_t * sigma_u)
        denominator = factor * bottom

        value = -0.5 * numerator / denominator * self.a_2(theta)

        return value

    def xi(self, theta):
        sigma_u = self.material.ultimate_tensile_strength
        epsilon_u = self.material.strain_at_uts

        l_cs = self.material.intrinsic_length

        numerator = 3 * epsilon_u * self.a_1(theta)
        denominator = sigma_u * l_cs ** 2

        value = (numerator / denominator) ** 0.5

        return value

    def eta(self, theta):
        sigma_u = self.material.ultimate_tensile_strength

        l_cs = self.material.intrinsic_length
        epsilon_u = self.material.strain_at_uts

        numerator = -3 * epsilon_u * self.a_2(theta)
        denominator = sigma_u * l_cs ** 2

        value = (numerator / denominator) ** 0.5

        return value

    def a_value(self, theta, thickness):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        factor = 1 / np.e ** (-self.xi(theta) * thickness)

        left_bracket = (4 * np.tan(theta)
                        - 9 * np.sin(2 * theta)
                        - 12 * np.power(np.sin(theta), 2)
                        - 6 * sigma_y * np.power(np.cos(theta), 2) * cot(theta) / sigma_u
                        - 2 * cot(theta)) / (48 * g)
        right_bracket = (sigma_y * cot(theta) * np.power(np.cos(theta), 2) / sigma_u
                         - np.power(np.cos(theta), 2) * cot(theta)) / (8 * g_t)

        value = factor * (left_bracket + right_bracket)

        return value

    def c_value(self, theta, thickness):
        g = self.material.elastic_shear_modulus()

        numerator = (np.sin(2 * theta) / (4 * g)
                     + self.a_value(theta, thickness) * np.e ** (-self.xi(theta) * thickness)
                     - self.b_2(theta) / self.a_2(theta))
        denominator = np.cos(self.eta(theta) * thickness)

        value = numerator / denominator

        return value


    def band_thickness_uniaxial(self):
        theta = np.deg2rad(np.arange(91))

        a_1 = self.a_1(theta)
        a_2 = self.a_2(theta)
        a_3 = self.a_3(theta)
        eta = self.eta(theta)

        trig = np.arctan((-a_1 / a_2) ** 0.5 * a_3)
        thickness = (np.pi + trig) / eta

        return thickness

    def strain_rate_11(self, theta):
        g = self.material.elastic_shear_modulus()

        fraction = 1 / (6 * g)
        brackets = 3 * np.power(np.sin(theta), 2) - 1

        value = fraction * brackets

        return value

    def strain_rate_12(self, theta, x2, thickness):
        g = self.material.elastic_shear_modulus()

        strain_rate_matrix = np.zeros((len(x2), len(theta)))

        for matrix_row, x2_value in enumerate(x2):
            for matrix_column, theta_value in enumerate(theta):

                thickness_value = thickness[matrix_column]

                if x2_value < thickness_value:
                    a = self.a_value(theta_value, thickness_value)
                    xi = self.xi(theta_value)
                    ic(a, xi)
                    strain_rate_matrix[matrix_row][matrix_column] = (np.sin(2 * theta_value) / (4 * g)
                                                                     + a * np.e ** (-xi * x2_value))
                else:
                    b_2 = self.b_2(theta_value)
                    a_2 = self.a_2(theta_value)
                    c = self.c_value(theta_value, thickness_value)
                    eta = self.eta(theta)

                    strain_rate_matrix[matrix_row][matrix_column] = (b_2 / a_2
                                                                     + c * np.cos(eta * x2_value))

        return strain_rate_matrix

    def strain_rate_22(self, theta, x2, thickness):
        g = self.material.elastic_shear_modulus()
        sigma_u = self.material.ultimate_tensile_strength
        epsilon_u = self.material.strain_at_uts

        strain_rate_21 = self.strain_rate_12(theta, x2, thickness)
        ratio = (sigma_u - 3 * g * epsilon_u) / epsilon_u

        strain_rate_11 = np.full((np.shape(strain_rate_21)), self.strain_rate_11(theta))

        top_left_bracket = (np.power(np.cos(theta), 2)
                            + strain_rate_21 * ratio * np.sin(2 * theta) * np.power(np.cos(theta), 2))
        top_right_bracket = strain_rate_11 * (2 * sigma_u / (3 * epsilon_u)
                                              - ratio * np.power(np.sin(2 * theta), 2))
        numerator = top_left_bracket - top_right_bracket

        denominator = (4 * sigma_u / (3 * epsilon_u)
                       - ratio * np.power(np.cos(2 * theta), 4))

        value = numerator / denominator

        return value

    def overall_effective_strain(self, theta, x2, thickness):
        term_one = self.strain_rate_12(theta, x2, thickness) * np.sin(2 * theta)
        term_two = np.full((np.shape(term_one)), self.strain_rate_11(theta)) * np.power(np.sin(theta), 2)
        term_three = self.strain_rate_22(theta, x2, thickness) * np.power(np.cos(theta), 2)

        value = term_one + term_two + term_three

        return value

    def plot_overall_effective_strain(self):
        theta = np.arange(19) * 5
        x2 = np.arange(11) * 10 ** -6
        thickness = self.band_thickness_uniaxial()

        overall_effective_strain = self.overall_effective_strain(theta, x2, thickness)
        print(overall_effective_strain)

    def thickness(self):
        """
        Based on the theory presented by S. Chen et al., "Prediction of the initial thickness of shear band localization
        based on a reduced strain gradient theory", 2011
        """

        if self.band_type == 'shear':
            return self.band_thickness_shear()

        elif self.band_type == 'uniaxial':
            return self.band_thickness_uniaxial()

        else:
            raise ValueError("Must be 'shear' or 'uniaxial'!")


def get_material_properties(name):
    materials = pd.read_csv(r'MaterialParametersShearBandThickness.csv').to_numpy()
    material_names = materials[:, 0]

    requested_position = np.where(material_names == name)
    properties = materials[requested_position][0][1:]

    return properties


# s235jr_steel = ShearBandMaterial(*get_material_properties("S235JR"))
# s235jr_band = ShearBand('shear', s235jr_steel)
# print(s235jr_band.thickness())
#
# ah36_steel = ShearBandMaterial(*get_material_properties("AH36"))
# ah36_band = ShearBand('shear', ah36_steel)
# print(ah36_band.thickness())

s355_steel = ShearBandMaterial(*get_material_properties("S355"))
s355_band = ShearBand('uniaxial', s355_steel)
s355_band.plot_overall_effective_strain()
