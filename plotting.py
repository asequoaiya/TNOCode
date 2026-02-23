# ----- Import libraries -----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'arial'


def load_maximum(x_data, y_data):
    clean_x_data = [x for x in x_data if not np.isnan(x)]
    clean_y_data = [y for y in y_data if not np.isnan(y)]

    maximum_y = np.amax(clean_y_data)
    maximum_y_position = np.argmax(clean_y_data)
    maximum_x = clean_x_data[maximum_y_position]

    return maximum_x, maximum_y


# ----- For mesh sensitivity -----
def mesh_sensitivity_plot():
    mesh_sensitivity_data = pd.read_csv("MeshSensitivity.csv", delimiter=";", header=0, index_col=0).to_numpy()
    maximum_force, maximum_extension, maximum_peeq, maximum_sigma, number_of_elements = mesh_sensitivity_data

    maximum_experimental_force = [50.300, 50.300]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(number_of_elements, maximum_force / 1000, "-o", label="ABAQUS numerical result")
    plt.plot([0, 30], maximum_experimental_force, "--", label="Experimental result")
    plt.legend()
    plt.ylabel("Maximum axial force [kN]")
    plt.xlabel("Number of through-thickness elements [-]")
    plt.title("Mesh sensitivity of recorded max. axial force")
    plt.xlim(0, 30)
    plt.ylim(50.2, 50.4)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(number_of_elements, maximum_peeq, "-o", label="ABAQUS numerical result")
    plt.legend()
    plt.ylabel("Maximum plastic equivalent plastic strain [-]")
    plt.xlabel("Number of through-thickness elements [-]")
    plt.title("Mesh sensitivity of max. plastic equivalent strain at max. force")
    plt.xlim(0, 30)
    plt.ylim(0.1, 0.13)
    plt.grid()
    plt.show()

    return


# ----- For mesh sensitivity -----
def time_sensitivity_plot():
    time_sensitivity_data = pd.read_csv("TimeSensitivity.csv", delimiter=";", header=0, index_col=0).to_numpy()
    maximum_force, maximum_extension, maximum_peeq, maximum_sigma, number_of_elements = time_sensitivity_data

    maximum_experimental_force = [50.300, 50.300]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(number_of_elements, maximum_force / 1000, "-o", label="ABAQUS numerical result")
    plt.plot([0, 0.03], maximum_experimental_force, "--", label="Experimental result")
    plt.legend()
    plt.ylabel("Maximum axial force [kN]")
    plt.xlabel("Maximum allowed time step [s]")
    plt.title("Time sensitivity of recorded max. axial force")
    plt.xlim(0, 0.03)
    plt.ylim(50.2, 50.4)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(number_of_elements, maximum_peeq, "-o", label="ABAQUS numerical result")
    plt.legend()
    plt.ylabel("Maximum plastic equivalent plastic strain [-]")
    plt.xlabel("Maximum allowed time step [s]")
    plt.title("Time sensitivity of max. plastic equivalent strain at max. force")
    plt.xlim(0, 0.03)
    plt.ylim(0.1, 0.13)
    plt.grid()
    plt.show()

    return


def hill_48_normal_plot():
    hill_48_normal_data = pd.read_csv("Hill48NormalData.csv", delimiter=",", header=0).to_numpy().transpose()
    (paper_extension, paper_stress, vm_extension, vm_stress,
     nguyen_extension, nguyen_stress, boudard_extension, boudard_stress) = hill_48_normal_data

    paper_max_x, paper_max_y = load_maximum(paper_extension, paper_stress)
    vm_max_x, vm_max_y = load_maximum(vm_extension, vm_stress)
    nguyen_max_x, nguyen_max_y = load_maximum(nguyen_extension, nguyen_stress)
    boudard_max_x, boudard_max_y = load_maximum(boudard_extension, boudard_stress)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(paper_extension, paper_stress / 1000000, "--", label="Experiment", color="tab:blue")
    plt.plot(paper_max_x, paper_max_y / 1000000, "-o", color="tab:blue")
    plt.plot(vm_extension, vm_stress / 1000000, label="von Mises", color="tab:orange")
    plt.plot(vm_max_x, vm_max_y / 1000000, "-o", color="tab:orange")
    plt.plot(nguyen_extension, nguyen_stress / 1000000, label="Hill '48 (Nguyen et al.)", color="tab:green")
    plt.plot(nguyen_max_x, nguyen_max_y / 1000000, "-o", color="tab:green")
    plt.plot(boudard_extension, boudard_stress / 1000000, label="Hill '48 (Revil-Boudard et al.)", color="tab:red")
    plt.plot(boudard_max_x, boudard_max_y / 1000000, "-o", color="tab:red")
    plt.legend()
    plt.ylabel("Nominal normal stress [MPa]")
    plt.xlabel("Extension [-]")
    plt.title(r"Influence of yield surface on nominal normal stress, $\alpha_n$ = 0.25")
    plt.xlim(0, 0.04)
    plt.ylim(0, 60)
    plt.grid()
    plt.show()

    return


def hill_48_shear_plot():
    hill_48_shear_data = pd.read_csv("Hill48ShearData.csv", delimiter=",", header=0).to_numpy().transpose()
    (paper_angle, paper_stress, vm_angle, vm_stress,
     nguyen_angle, nguyen_stress, boudard_angle, boudard_stress) = hill_48_shear_data

    paper_max_x, paper_max_y = load_maximum(paper_angle, paper_stress)
    vm_max_x, vm_max_y = load_maximum(vm_angle, vm_stress)
    nguyen_max_x, nguyen_max_y = load_maximum(nguyen_angle, nguyen_stress)
    boudard_max_x, boudard_max_y = load_maximum(boudard_angle, boudard_stress)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(paper_angle, paper_stress / 1000000, "--", label="Experiment", color="tab:blue")
    plt.plot(paper_max_x, paper_max_y / 1000000, "-o", color="tab:blue")
    plt.plot(vm_angle, vm_stress / 1000000, label="von Mises", color="tab:orange")
    plt.plot(vm_max_x, vm_max_y / 1000000, "-o", color="tab:orange")
    plt.plot(nguyen_angle, nguyen_stress / 1000000, label="Hill '48 (Nguyen et al.)", color="tab:green")
    plt.plot(nguyen_max_x, nguyen_max_y / 1000000, "-o", color="tab:green")
    plt.plot(boudard_angle, boudard_stress / 1000000, label="Hill '48 (Revil-Boudard et al.)", color="tab:red")
    plt.plot(boudard_max_x, boudard_max_y / 1000000, "-o", color="tab:red")
    plt.legend()
    plt.ylabel("Nominal shear stress [MPa]")
    plt.xlabel("Twist angle [deg]")
    plt.title(r"Influence of yield surface on nominal shear stress, $\alpha_n$ = 0.25")
    plt.xlim(0, 40)
    plt.ylim(0, 200)
    plt.grid()
    plt.show()

    return


def hill_48_ps_plot():
    hill_48_normal_data = pd.read_csv("Hill48NormalDataPS.csv", delimiter=",", header=0).to_numpy().transpose()
    (paper_extension, paper_stress, vm_extension, vm_stress,
     nguyen_extension, nguyen_stress, boudard_extension, boudard_stress) = hill_48_normal_data

    paper_max_x, paper_max_y = load_maximum(paper_extension, paper_stress)
    vm_max_x, vm_max_y = load_maximum(vm_extension, vm_stress)
    nguyen_max_x, nguyen_max_y = load_maximum(nguyen_extension, nguyen_stress)
    boudard_max_x, boudard_max_y = load_maximum(boudard_extension, boudard_stress)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(paper_extension, paper_stress / 1000000, "--", label="Experiment", color="tab:blue")
    plt.plot(paper_max_x, paper_max_y / 1000000, "-o", color="tab:blue")
    plt.plot(vm_extension, vm_stress / 1000000, label="von Mises", color="tab:orange")
    plt.plot(vm_max_x, vm_max_y / 1000000, "-o", color="tab:orange")
    plt.plot(nguyen_extension, nguyen_stress / 1000000, label="Hill '48 (Nguyen et al.)", color="tab:green")
    plt.plot(nguyen_max_x, nguyen_max_y / 1000000, "-o", color="tab:green")
    plt.plot(boudard_extension, boudard_stress / 1000000, label="Hill '48 (Revil-Boudard et al.)", color="tab:red")
    plt.plot(boudard_max_x, boudard_max_y / 1000000, "-o", color="tab:red")
    plt.legend()
    plt.ylabel("Nominal normal stress [MPa]")
    plt.xlabel("Extension [-]")
    plt.title(r"Influence of yield surface on nominal normal stress, plane strain tension")
    plt.xlim(0, 0.1)
    plt.ylim(0, 400)
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(paper_extension, paper_stress / 1000000, "-x", label="Experiment", color="tab:blue")
    # plt.plot(paper_max_x, paper_max_y / 1000000, "-o", color="tab:blue")
    plt.plot(vm_extension, vm_stress / 1000000, label="von Mises", color="tab:orange")
    plt.plot(vm_max_x, vm_max_y / 1000000, "-o", color="tab:orange")
    plt.plot(nguyen_extension, nguyen_stress / 1000000, label="Hill '48 (Nguyen et al.)", color="tab:green")
    plt.plot(nguyen_max_x, nguyen_max_y / 1000000, "-o", color="tab:green")
    plt.plot(boudard_extension, boudard_stress / 1000000, label="Hill '48 (Revil-Boudard et al.)", color="tab:red")
    plt.plot(boudard_max_x, boudard_max_y / 1000000, "-o", color="tab:red")
    plt.legend()
    plt.ylabel("Nominal normal stress [MPa]")
    plt.xlabel("Extension [-]")
    plt.title(r"Influence of yield surface on nominal normal stress, plane strain tension, zoomed in")
    plt.xlim(0, 0.1)
    plt.ylim(320, 360)
    plt.grid()
    plt.show()


def material_calibration_plot():
    material_calibration_data = pd.read_csv("MaterialCalibrations.csv", delimiter=",", header=0).to_numpy().transpose()
    plastic_strain, vm_stress, hill48_stress = material_calibration_data

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(plastic_strain, vm_stress / 1000000, label="von Mises", color="tab:blue")
    plt.plot(plastic_strain, hill48_stress / 1000000, label="Hill '48", color="tab:orange")
    plt.legend()
    plt.ylabel("Stress [MPa]")
    plt.xlabel("Plastic strain [-]")
    plt.title(r"Material calibration curves")
    plt.xlim(0, 1)
    plt.ylim(0, 400)
    plt.grid()
    plt.show()


# hill_48_normal_plot()
# hill_48_shear_plot()
material_calibration_plot()