import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
alpha = 3.85e-3
R_0 = 100
r = 39
V_c = 30

R_1 = r * R_0
R_2 = r * R_0
R_3 = R_0

S = alpha * R_0
S_L = r * alpha * V_c / (r + 1) ** 2
G = 5 / (100 * S_L)

# Modelo Ideal
sl = S_L * 1000


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
    return abs((param[0] - const)) / const


# DATA PREP
data = read_data("data_1.txt")
t_testigo = data[0]
t_medida = data[3]
v_s = data[1]
v_a = data[2]

# ------------------------------------------------
#      ----------------BRIDGE----------------
# ------------------------------------------------
print("\n\n\nBRIDGE")
t_theoretical = np.linspace(0, 100, len(t_testigo))

# Curve Fitted
param_bri, pcov_bri = curve_fit(linear, t_testigo, v_a)
err_bri = np.sqrt(np.diag(pcov_bri))

v_a_fitting = linear(t_theoretical, param_bri[0], param_bri[1])
bri_equation = f"V_sl[mV] = ({param_bri[0]:.3f} ± {err_bri[0]:.2}) * t + ({param_bri[1]:.2f} ± {err_bri[1]:.2})"
print(bri_equation)

# Model
v_a_model = linear(t_theoretical, sl, 0)

# ERRORS
zero_err_bri = zero_error(param_bri)
sensitivity_err_bri = sensitivity_error(param_bri, sl) * 100
lineal_err_bri = lineal_error(v_a, t_testigo, param_bri)

str_zero_bri = f"Error de cero: \u03B5 = {zero_err_bri:.2f}"
str_sens_bri = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_bri:.2f}%"
str_lineal_bri = f"Error de linealidad (%): \u03B5 = {lineal_err_bri:.2f}%"

print(str_zero_bri)
print(str_sens_bri)
print(str_lineal_bri)

# Plot
plt.plot(t_theoretical, v_a_fitting, "b", label="Curve fitted")
plt.plot(t_testigo, v_a, "r*", label="Experimental data")
plt.plot(t_theoretical, v_a_model, "k", label="Theoretical Model")
plt.legend()
plt.grid()
plt.title("Bridge Analysis")
plt.xlabel("T ($^{\circ}$C)")
plt.ylabel("$V_{sl}$ (mV)")
plt.savefig("bridge_rtd")
plt.show()

# ------------------------------------------
#     ----------------AMP----------------
# ------------------------------------------
print("\n\n\nAMPLIFIER")
v_a_theoretical = np.linspace(0, sl * 100, len(v_a))

# Curve Fitted
param_amp, pcov_amp = curve_fit(linear, v_a, v_s)
err_amp = np.sqrt(np.diag(pcov_amp))
print(err_amp)
v_o_fittingV = linear(v_a_theoretical, param_amp[0], param_amp[1])
amp_equation = f"V_0[V] = ({param_amp[0]:.6f} ± {err_amp[0]:.2}) * V_sl + ({param_amp[1]:.4f} ± {err_amp[1]:.2})"
print(amp_equation)

# Model
v_o_modelV = linear(v_a_theoretical, G / 1000, 0)

# ERRORS
zero_err_amp = zero_error(param_amp)
sensitivity_err_amp = sensitivity_error(param_amp, G * 1000)
lineal_err_amp = lineal_error(v_s, v_a, param_amp)

str_zero_amp = f"Error de cero: \u03B5 = {zero_err_amp:.2f}"
str_sens_amp = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_amp:.2f}%"
str_lineal_amp = f"Error de linealidad (%): \u03B5 = {lineal_err_amp:.2f}%"

print(str_zero_amp)
print(str_sens_amp)
print(str_lineal_amp)

# Plot
plt.plot(v_a_theoretical, v_o_fittingV, "b", label="Curve fitted")
plt.plot(v_a, v_s, "r*", label="Experimental data")
plt.plot(v_a_theoretical, v_o_modelV, "k", label="Theoretical Model")
plt.legend()
plt.grid()
plt.title("Amplifier Analysis")
plt.xlabel("$V_{sl}$ (mV)")
plt.ylabel("$V_0$ (V)")
plt.savefig("amplifier_rtd")
plt.show()

# ------------------------------------------------
#      ----------------SYSTEM----------------
# ------------------------------------------------
print("\n\n\nSYSTEM")
# Curve Fitted
param_sys, pcov_sys = curve_fit(linear, t_testigo, v_s)
err_sys = np.sqrt(np.diag(pcov_sys))

v_o_fittingT = linear(t_theoretical, param_sys[0], param_sys[1])
sys_equation = f"V_0[V] = ({param_sys[0]:.4f} ± {err_sys[0]:.2}) * t + ({param_sys[1]:.2f} ± {err_sys[1]:.2})"
print(sys_equation)

# Model
v_o_modelT = linear(t_theoretical, G * sl / 1000, 0)

# ERRORS
zero_err_sys = zero_error(param_sys)
sensitivity_err_sys = sensitivity_error(param_sys, sl)
lineal_err_sys = lineal_error(v_s, t_testigo, param_sys)

str_zero_sys = f"Error de cero: \u03B5 = {zero_err_sys:.2f}"
str_sens_sys = f"Error de sensibilidad(%): \u03B5 = {sensitivity_err_sys:.2f}%"
str_lineal_sys = f"Error de linealidad (%): \u03B5 = {lineal_err_sys:.2f}%"

print(str_zero_sys)
print(str_sens_sys)
print(str_lineal_sys)

# Plot
plt.plot(t_theoretical, v_o_fittingT, "b", label="Curve fitted")
plt.plot(t_testigo, v_s, "r*", label="Experimental data")
plt.plot(t_theoretical, v_o_modelT, "k", label="Theoretical Model")
plt.legend()
plt.grid()
plt.title("System Analysis")
plt.xlabel("T ($^{\circ}$C)")
plt.ylabel("$V_0$ (V)")
plt.savefig("system_rtd")
plt.show()

# ------------------------------------------------
#   ----------------TEMPERATURES----------------
# ------------------------------------------------
print("\n\n\nTEMPERATURE")
# Curve Fitted
param_temp, pcov_temp = curve_fit(linear, t_testigo, t_medida)
err_temp = np.sqrt(np.diag(pcov_temp))

t_fitting = linear(t_theoretical, param_temp[0], param_temp[1])
temp_equation = f"t_med[ºC] = ({param_temp[0]:.4f} ± {err_temp[0]:.2}) * t_test + ({param_temp[1]:.2f} ± {err_temp[1]:.2})"
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
plt.plot(t_theoretical, t_theoretical_model, "k", label="Theoretical Model")
plt.legend()
plt.grid()
plt.title("Temperature")
plt.xlabel("T thermometer ($^{\circ}$C)")
plt.ylabel("T RTD ($^{\circ}$C)")
plt.savefig("temperatures_rtd")
plt.show()
# ------------------------------------------------------------- #
# -------------------------- To .txt -------------------------- #
# ------------------------------------------------------------- #

file_name = "results_rtd"
check = file_name
i = 1
while os.path.isfile(check + ".txt"):
    i += 1
    check = "%s_%d" % (file_name, i)
if i > 1:
    file_name = check

f = open(file_name + ".txt", "w")
f.writelines("\t\t\t\t\t+--------------+\n")
f.writelines("\t\t\t\t\t|  RTD SENSOR  |\n")
f.writelines("\t\t\t\t\t+--------------+\n")
f.writelines("\n\n")
f.writelines("+---------------------BRIDGE---------------------+\n")
f.writelines("  {} \n".format(bri_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Errors:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Zero drift:", "\u03B5", zero_err_bri, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity drift:", "\u03B5", sensitivity_err_bri, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Linearity:", "\u03B5", lineal_err_bri, "2f", "%"
    )
)
f.writelines("+-------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+------------------------AMPLIFIER-------------------------+\n")
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
f.writelines("+----------------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+---------------------SYSTEM---------------------+\n")
f.writelines("  {} \n".format(sys_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Errors:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Zero drift:", "\u03B5", zero_err_sys, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity drift:", "\u03B5", sensitivity_err_sys, "2f", "%"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Linearity:", "\u03B5", lineal_err_sys, "2f", "%"
    )
)
f.writelines("+-------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+----------------------TEMPERATURE----------------------+\n")
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
f.writelines("+-----------------------------------------------------+\n")
f.close()
