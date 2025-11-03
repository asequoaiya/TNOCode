# ----- Import libraries -----
import numpy as np
import matplotlib.pyplot as plt

# Bressan-Williams for pure shear
def pure_shear_bressan_williams(strain_ratio, critical_shear_stress):
    fraction = (1 + strain_ratio) / (1 - strain_ratio)
    denominator = (1 - fraction ** 2) ** 0.5

    sigma_1 = critical_shear_stress / denominator

    return sigma_1

# Bressan-Williams for surrounding shear
def surrounding_shear_bressan_williams(strain_ratio, critical_shear_stress):
    stress_ratio = (2 * strain_ratio + 1) / (2 + strain_ratio)
    brackets = 1 + stress_ratio

    fraction = (1 + strain_ratio) / (-1 + strain_ratio)
    root = (1 - fraction ** 2) ** 0.5

    denominator = brackets * root

    sigma_1 = critical_shear_stress / denominator

    return sigma_1

def hill_local_necking(strain_ratio, power_law_amplitude, power_law_exponent):
    left_fraction = 2 * power_law_amplitude / (3 ** 0.5)

    middle_top = 1 + 0.5 * strain_ratio
    polynomial = (np.power(strain_ratio, 2) + strain_ratio + 1) ** 0.5
    middle_fraction = middle_top / polynomial

    bracket_left = 2 / (3 ** 0.5)
    bracket_middle = power_law_exponent / (1 + strain_ratio)
    brackets = (bracket_left * bracket_middle * polynomial) ** power_law_exponent

    sigma_1 = left_fraction * middle_fraction * brackets

    return sigma_1

def calibration_shear_stress(power_law_amplitude, power_law_exponent):
    argument = (2 * power_law_exponent) ** power_law_exponent
    factor = 2 * (2 ** 0.5) * power_law_amplitude / 3

    critical = factor * argument

    return critical


beta_values = np.linspace(-1.5, -0.5, 101)
beta_hill = np.linspace(-0.5, 0, 51)

calibrated_shear_stress = calibration_shear_stress(1, 0.1)

pure_shear = pure_shear_bressan_williams(beta_values, calibrated_shear_stress)
surrounding_shear = surrounding_shear_bressan_williams(beta_values, calibrated_shear_stress)
hill_results = hill_local_necking(beta_hill, 1, 0.1)

plt.plot(beta_values, pure_shear, label="Pure shear")
plt.plot(beta_values, surrounding_shear, label="Surrounding shear")
plt.plot(beta_hill, hill_results, label="Hill local necking")
plt.xlabel("Strain ratio (beta)")
plt.ylabel("Normalized sigma 1")
plt.legend()
plt.grid()
plt.show()
