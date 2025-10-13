# ----- Import libraries -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        g_s = self.material.softening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        l_cs = self.material.intrinsic_length

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

        l_cs = self.material.intrinsic_length

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

    def a_3(self, theta):
        g = self.material.elastic_shear_modulus()
        g_t = self.material.hardening_shear_modulus()
        g_s = self.material.softening_shear_modulus()

        sigma_y = self.material.yield_strength
        sigma_u = self.material.ultimate_tensile_strength

        l_cs = self.material.intrinsic_length

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

    def eta(self, theta):
        sigma_u = self.material.ultimate_tensile_strength

        l_cs = self.material.intrinsic_length
        epsilon_u = self.material.strain_at_uts

        numerator = -3 * epsilon_u * self.a_2(theta)
        denominator = sigma_u * l_cs ** 2

        value = (numerator / denominator) ** 0.5

        return value

    def band_thickness_uniaxial(self):
        theta = np.deg2rad(np.arange(91))

        a_1 = self.a_1(theta)
        a_2 = self.a_2(theta)
        a_3 = self.a_3(theta)
        eta = self.eta(theta)


        trig = np.arctan((-a_1 / a_2) ** 0.5 * a_3)
        thickness = (np.pi + trig) / eta

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


s235jr_steel = ShearBandMaterial(*get_material_properties("S235JR"))
s235jr_band = ShearBand('shear', s235jr_steel)
print(s235jr_band.thickness())

ah36_steel = ShearBandMaterial(*get_material_properties("AH36"))
ah36_band = ShearBand('shear', ah36_steel)
print(ah36_band.thickness())

s355_steel = ShearBandMaterial(*get_material_properties("S355"))
s355_band = ShearBand('uniaxial', s355_steel)
print(s355_band.thickness())
