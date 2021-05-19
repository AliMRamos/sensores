import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# DATA
"""
V_0 = (V_Tc,T0 +16microV)193.4 -> V_Tc,0 = V_0/193.4 - 16 microV
V_0 = 250 mV @ 25ºC
"""
G = 193.4
v_at_25 = 250  # mV
V_25_0 = 1.277  # mV
s_model1 = 52.2  # µV/ºC
s_model2 = 51.7  # µV/ºC
S_AD594 = 10  # mV/ºC


def read_data(file_data):
    data = np.loadtxt(file_data, skiprows=1).T
    return data[0], data[1], data[2], data[3]


def linear(x, a, b):
    return a * x + b


def lineal_error(real_data_y, real_data_x, param):
    h = max(abs(real_data_y - linear(real_data_x, param[0], param[1])))
    fso = max(real_data_y) - min(real_data_y)
    error_lin = 100 * h / fso
    return error_lin


def zero_error(param):
    return abs(param[1] / param[0])


def sensitivity_error(param, const):
    return 100 * abs((param[0] - const)) / const


# DATA PREP
data = read_data("data_3.txt")
t_testigo = data[0]  # T_c
t_medida = data[3]
v_tc = data[1]
v_out = data[2]

# ------------------------------------------------------
#      ----------------THERMOCOUPLE----------------
# ------------------------------------------------------
print("\n\n\nTHERMOCOUPLE")
t_theoretical = np.linspace(0, 100, len(t_testigo))

# Curve Fitted
param_thermo, pcov_thermo = curve_fit(linear, t_testigo, v_tc)
err_thermo = np.sqrt(np.diag(pcov_thermo))

v_tc_fitting = linear(t_theoretical, param_thermo[0], param_thermo[1])
thermo_equation = f"V_Tc,0[mV] = ({param_thermo[0]:.4f} ± {err_thermo[0]:.2}) * t + ({param_thermo[1]:.3f} ± {err_thermo[1]:.2})"
print(thermo_equation)

# Models
v_tc_model_1 = linear(t_theoretical, s_model1 / 1000, 0)
v_tc_model_2 = linear(t_theoretical, s_model2 / 1000, 0)

# ERRORS
zero_err_thermo = zero_error(param_thermo)
sensitivity_err_thermo_1 = sensitivity_error(param_thermo, s_model1 / 1000)
sensitivity_err_thermo_2 = sensitivity_error(param_thermo, s_model2 / 1000)
lineal_err_thermo = lineal_error(v_tc, t_testigo, param_thermo)

str_zero_thermo = f"Error de cero: \u03B5 = {zero_err_thermo:.2f}"
str_sens_thermo_1 = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_thermo_1:.2f}% (Model 1)"
str_sens_thermo_2 = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_thermo_2:.2f}% (Model 2)"
str_lineal_thermo = f"Error de linealidad (%): \u03B5 = {lineal_err_thermo:.2f}%"

print(str_zero_thermo)
print(str_sens_thermo_1)
print(str_sens_thermo_2)
print(str_lineal_thermo)

# Plot
# plt.plot([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#         [0.000, 0.507, 1.019, 1.536, 2.058, 2.585, 3.115, 3.649, 4.186, 4.725, 5.268], "oc", label="Tabla")
plt.plot(t_theoretical, v_tc_fitting, color="orange", label="Curve fitted")
plt.plot(t_testigo, v_tc, "r*", label="Experimental data")
plt.plot(t_theoretical, v_tc_model_1, ":b", label="Model 1 (S=" + str(s_model1) + ")")
plt.plot(t_theoretical, v_tc_model_2, "--g", label="Model 2 (S=" + str(s_model2) + ")")
plt.legend()
plt.grid()
plt.title("Thermocouple Analysis")
plt.xlabel("$T_{c}$ ($^{\circ}$C)")
plt.ylabel("$V_{Tc,0}$ (mV)")
plt.savefig("thermocouple.png")
plt.show()

# ------------------------------------------
#     ----------------AMP----------------
# ------------------------------------------
print("\n\n\nAMPLIFIER")
v_a_theoretical = np.linspace(0, 1, len(v_out))

# Curve Fitted
param_amp, pcov_amp = curve_fit(linear, t_testigo, v_out)
err_amp = np.sqrt(np.diag(pcov_amp))

v_o_fittingV = linear(t_theoretical, param_amp[0], param_amp[1])

amp_equation = f"V_0[V] = ({param_amp[0]:.5f} ± {err_amp[0]:.2}) * t_c + ({param_amp[1]:.3f} ± {err_amp[1]:.2})"
print(amp_equation)

# Model
v_o_modelV = linear(t_theoretical, S_AD594 / 1000, 0)

# ERRORS
zero_err_amp = zero_error(param_amp)
sensitivity_err_amp = sensitivity_error(param_amp, S_AD594 / 1000)
lineal_err_amp = lineal_error(v_out, t_testigo, param_amp)

str_zero_amp = f"Error de cero: \u03B5 = {zero_err_amp:.2f}"
str_sens_amp = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_amp:.2f}%"
str_lineal_amp = f"Error de linealidad (%): \u03B5 = {lineal_err_amp:.2f}%"

print(str_zero_amp)
print(str_sens_amp)
print(str_lineal_amp)

# Plot
plt.plot(t_theoretical, v_o_fittingV, "b", label="Curve fitted")
plt.plot(t_testigo, v_out, "r*", label="Experimental data")
plt.plot(t_theoretical, v_o_modelV, color="orange", label="Theoretical Model")
# plt.hlines(0.250, 0, 100)
# plt.vlines(25, 0, 1)
plt.legend()
plt.grid()
plt.title("Amplifier Analysis")
plt.xlabel("$T_{c}$ ($^{\circ}$C)")
plt.ylabel("$V_0$ (V)")
plt.savefig("amplifier_thermocouple")
plt.show()

# ------------------------------------------------
#   ----------------TEMPERATURES----------------
# ------------------------------------------------
print("\n\n\nTEMPERATURE")
# Curve Fitted
param_temp, pcov_temp = curve_fit(linear, t_testigo, t_medida)
err_temp = np.sqrt(np.diag(pcov_temp))

t_fitting = linear(t_theoretical, param_temp[0], param_temp[1])
temp_equation = f"t_med[ºC] = ({param_temp[0]:.3f} ± {err_temp[0]:.2}) * t_test + ({param_temp[1]:.2f} ± {err_temp[1]:.2})"
print(temp_equation)

# Model
t_theoretical_model = linear(t_theoretical, 1, 0)

# ERRORS
zero_err_temp = zero_error(param_temp)
sensitivity_err_temp = sensitivity_error(param_temp, 1)
lineal_err_temp = lineal_error(t_medida, t_testigo, param_temp)

str_zero_temp = f"Error de cero: \u03B5 = {zero_err_temp:.2f}"
str_sens_temp = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_temp:.2f}%"
str_lineal_temp = f"Error de linealidad (%): \u03B5 = {lineal_err_temp:.2f}%"

print(str_zero_temp)
print(str_sens_temp)
print(str_lineal_temp)

# Plot
plt.plot(t_theoretical, t_fitting, "b", label="Curve fitted")
plt.plot(t_testigo, t_medida, "r*", label="Experimental data")
plt.plot(t_theoretical, t_theoretical_model, color="orange", label="Theoretical Model")
plt.legend()
plt.grid()
plt.title("Temperature")
plt.xlabel("T thermometer ($^{\circ}$C)")
plt.ylabel("T thermocuple ($^{\circ}$C)")
plt.savefig("temperatures_thermocouple")
plt.show()

# ------------------------------------------------------------- #
# -------------------------- To .txt -------------------------- #
# ------------------------------------------------------------- #

file_name = "results_thermocouple"
check = file_name
i = 1
while os.path.isfile(check + ".txt"):
    i += 1
    check = "%s_%d" % (file_name, i)
if i > 1:
    file_name = check

f = open(file_name + ".txt", "w")
f.writelines("\t\t\t\t\t+-----------------------+\n")
f.writelines("\t\t\t\t\t|  THERMOCOUPLE SENSOR  |\n")
f.writelines("\t\t\t\t\t+-----------------------+\n")
f.writelines("\n\n")
f.writelines("+---------------------THERMOCOUPLE---------------------+\n")
f.writelines("  {} \n".format(thermo_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Errors:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Zero drift:", "\u03B5", zero_err_thermo, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity drift:", "\u03B5", sensitivity_err_thermo_1, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Linearity:", "\u03B5", lineal_err_thermo, "2f", "%"
    )
)
f.writelines("+-----------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+----------------------AMPLIFIER-----------------------+\n")
f.writelines("  {} \n".format(amp_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Errors:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Zero drift:", "\u03B5", zero_err_amp, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity drift:", "\u03B5", sensitivity_err_amp, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Linearity:", "\u03B5", lineal_err_amp, "2f", "%"
    )
)
f.writelines("+------------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+---------------------TEMPERATURE---------------------+\n")
f.writelines("  {} \n".format(temp_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Errors:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Zero drift:", "\u03B5", zero_err_temp, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity drift:", "\u03B5", sensitivity_err_temp, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Linearity:", "\u03B5", lineal_err_temp, "2f", "%"
    )
)
f.writelines("+---------------------------------------------------+\n")
f.close()
