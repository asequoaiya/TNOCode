# ----- Import libraries -----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'arial'

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


mesh_sensitivity_plot()
time_sensitivity_plot()