# ----- Import libraries -----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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


def alpha_results_loop(function):
    # Loop through all files in the AlphaResults directory
    for entry in os.scandir("AlphaResults"):
        alpha_results_plot(entry.path, str(entry)[16:-6], function)


def alpha_results_plot(path, case, function):
    # Read in .csv data as pandas DataFrame
    data = pd.read_csv(path, delimiter=";")
    alpha_dictionary = {"30":3.0, "25":2.5, "20":2.0, "15":1.5, "125":1.25, "10":1.0, "075":0.75, "05":0.5,
                        "0375":0.375, "025":0.25, "PS":"PS"}


    # Extract data
    time = (data["Time [s]"].dropna()).to_numpy()
    force = (data["RF Mag [N]"].dropna()).to_numpy()

    sigma = (data["Sigma [MPa]"].dropna()).to_numpy()
    extension = (data["Extension [-]"].dropna()).to_numpy()

    paper_extension = (data["Paper extension [-]"].dropna()).to_numpy()
    paper_sigma = (data["Paper Sigma [MPa]"].dropna()).to_numpy()

    if type(alpha_dictionary[case]) == float:
        moment = (abs(data["RM2 [Nm]"].dropna()).to_numpy())
        tau = (data["Tau [MPa]"].dropna()).to_numpy()
        angle = (data["Actual angle [deg]"].dropna()).to_numpy()
        paper_angle = (data["Paper angle [deg]"].dropna()).to_numpy()
        paper_tau = (data["Paper Tau [MPa]"].dropna()).to_numpy()

        # Constants
        base_factor = 45.053
        loading_factor = base_factor * alpha_dictionary[case]

        # --- Moment-force relation curve ---
        ideal_y = [0, np.amax(moment) * loading_factor]
        ideal_x = [0, np.amax(moment)]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(moment, force, label="ABAQUS")
        plt.plot(ideal_x, ideal_y, "--", label="Ideal")
        plt.title(fr"Force-moment relation curve for $\alpha_n$ = {alpha_dictionary[case]}")
        plt.ylabel("Resultant force [N]")
        plt.xlabel("Resultant moment [Nm]")
        plt.ylim(0, 1.1 * np.amax(moment) * loading_factor)
        plt.xlim(0, 1.1 * np.amax(moment))
        plt.grid()
        plt.legend()

        if function == "show":
            plt.show()
        elif function == "save":
            plt.savefig(f"AlphaFigures/MFRelationAlpha{case}.png")

        # --- Loading ratio plot ---
        loading_ratio = force[2:] / moment[2:]
        ideal_loading_ratio_x = [0, np.amax(time)]
        ideal_loading_ratio_y = [loading_factor, loading_factor]

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(time[2:], loading_ratio, label="ABAQUS")
        plt.plot(ideal_loading_ratio_x, ideal_loading_ratio_y, "--", label="Ideal")
        plt.title(fr"Loading ratio for $\alpha_n$ = {alpha_dictionary[case]}")
        plt.ylabel("Loading ratio [m$^{-1}$]")
        plt.xlabel("ABAQUS time [s]")
        plt.ylim(0.95 * loading_factor, 1.05 * loading_factor)
        plt.xlim(0, np.amax(time))
        plt.grid()
        plt.legend()

        if function == "show":
            plt.show()
        elif function == "save":
            plt.savefig(f"AlphaFigures/LoadingRatioAlpha{case}.png")

        # --- Nominal shear stress ---
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(angle, tau, label="ABAQUS")
        plt.plot(paper_angle, paper_tau, "--", label="Ideal")
        plt.title(fr"Nominal shear stress for $\alpha_n$ = {alpha_dictionary[case]}")
        plt.ylabel("Nominal shear stress [MPa]")
        plt.xlabel("Twist angle [deg]")
        plt.ylim(0, 1.1 * np.amax(tau))
        plt.xlim(0, 1.1 * np.amax(angle))
        plt.grid()
        plt.legend()

        if function == "show":
            plt.show()
        elif function == "save":
            plt.savefig(f"AlphaFigures/ShearAlpha{case}.png")

    # --- Nominal shear stress ---
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(extension, sigma, label="ABAQUS")
    plt.plot(paper_extension, paper_sigma, "--", label="Ideal")
    plt.title(fr"Nominal normal stress for $\alpha_n$ = {alpha_dictionary[case]}")
    plt.ylabel("Nominal normal stress [MPa]")
    plt.xlabel("Extension [-]")
    plt.ylim(0, 1.1 * np.amax(sigma))
    plt.xlim(0, 1.1 * np.amax(extension))
    plt.grid()
    plt.legend()

    if function == "show":
        plt.show()
    elif function == "save":
        plt.savefig(f"AlphaFigures/NormalAlpha{case}.png")


def alpha_relation_plot():
    alpha_relation_data = pd.read_csv("AlphaRelation.csv", sep=";").to_numpy().transpose()
    alpha, time, phi_radians, phi_degrees, phi_alpha = alpha_relation_data

    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(alpha, phi_radians, marker="x", label="Exact")
    plt.scatter(alpha, phi_alpha, marker="+", label="Approximation")
    plt.title(r"Comparison of principal rotation angle $\varphi$ with and without approximation")
    plt.ylabel(r"Principal rotation angle $\varphi$ [rad]")
    plt.xlabel(r"Loading ratio $\alpha_n$ [-]")
    plt.xlim(0, 1.1 * np.amax(alpha))
    plt.ylim(0, 1.1 * np.amax((np.amax(phi_radians), np.amax(phi_alpha))))
    plt.grid()
    plt.legend()
    plt.show()


# hill_48_normal_plot()
# hill_48_shear_plot()
alpha_relation_plot()
