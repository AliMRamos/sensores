import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
t_0 = 25 + 273.15
t_c = 60 + 273.15
dt_max = 0.1
delta = 1E-2


def ntc_data(file_data):
    data = np.loadtxt(file_data, skiprows=1).T
    abs_data_temp = [i + 273.15 for i in data[0]]
    return abs_data_temp, data[1]


def read_data(file_data):
    data = np.loadtxt(file_data, skiprows=1).T
    return data[0], data[1], data[2], data[3]


def curve_ntc(t, r_0, b, t_ini=t_0):
    return r_0 * np.exp(b * ((1 / t) - (1 / t_ini)))


def real_ntc(t, vi, r, b, r_0, t0=t_0):
    return vi * r / (r + curve_ntc(t, r_0, b, t0))


def linear(x, a, b):
    return a * x + b


def linear_ntc(t, vi, r, b, r_0):
    return real_ntc(t_c, vi, r, b, r_0) + (vi / b) * (b ** 2 / (4 * t_c ** 2) - 1) * (t - t_c)


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
data = ntc_data("dataLab.txt")
r_exp = data[1]
t_exp = data[0]

# ----------------------------------------------------------
#      ----------------CHARACTERIZATION----------------
# ----------------------------------------------------------
print("\n\nCHARACTERIZACION")
t_teor = np.linspace(min(t_exp), max(t_exp), 100)

# Curve Fitted
param_ntc, pcov_ntc = curve_fit(curve_ntc, t_exp, r_exp, p0=[20, 1], absolute_sigma=True)
err_ntc = np.sqrt(np.diag(pcov_ntc))

r_fitting = curve_ntc(t_teor, param_ntc[0], param_ntc[1])
ntc_equation = f"R = ({param_ntc[0]:.2f} ± {err_ntc[0]:.2})*e^[({param_ntc[1]:.2f}= ± {err_ntc[1]:.2e})*(1/T-1/{t_0})]"
print(ntc_equation)

# Plot
plt.plot(t_exp, r_exp, "r*", label="Experimental data")
plt.plot(t_teor, r_fitting, "b", label="Curve fitted")
plt.grid()
plt.xlabel("T (K)")
plt.ylabel("R (K$\Omega$)")
plt.title("NTC behaviour")
plt.legend()
plt.savefig("ntc_behaviour")
plt.show()

# NEW DATA
r_constant = param_ntc[0]
b_constant = param_ntc[1]

r_design = (b_constant - 2 * t_c) / (b_constant + 2 * t_c) * curve_ntc(t_c, r_constant, b_constant)
print(f"R design: {r_design:.2f} kΩ")
v_i_max = 2 * np.sqrt(delta * r_design * dt_max * 1E3)
print(f"Vi max: {v_i_max:.2f} V")

t_model = np.linspace(10, 110, 1000) + 273.15

plt.figure()
plt.plot(t_model,
         linear_ntc(t_model, v_i_max, r_design, b_constant, r_constant), "-b", label="Lineal NTC")
plt.plot(t_model, real_ntc(t_model, v_i_max, r_design, b_constant, r_constant), "-r", label="Real NTC")
plt.vlines(30 + 273.15, 0, 2.3, 'k', linestyles='--')
plt.vlines(90 + 273.15, 0, 2.3, 'k', linestyles='--')
plt.title("Linear behaviour of the NTC")
plt.xlabel("T (K)")
plt.ylabel("V (V)")
plt.grid()
plt.legend(loc="upper center")
plt.savefig("linear_ntc", dpi=1500)
plt.show()

# ERRORS
t_range = np.linspace(30, 90, 1000) + 273.15
h = max(abs(
    real_ntc(t_range, v_i_max, r_design, b_constant, r_constant) - linear_ntc(t_range, v_i_max, r_design, b_constant,
                                                                              r_constant)))
lineal_err_ntc = h / (
        linear_ntc(90 + 273.15, v_i_max, r_design, b_constant, r_constant) - linear_ntc(30 + 273.15, v_i_max,
                                                                                        r_design,
                                                                                        b_constant,
                                                                                        r_constant)) * 100
str_lineal_ntc = f"Error de linealidad (%): \u03B5 = {lineal_err_ntc:.2f}%"
print(f"--Lineal behaviour--\n{str_lineal_ntc}")

v_off = v_i_max * r_design / (r_design + curve_ntc(30 + 273.15, r_constant, b_constant))
sl = (v_i_max / b_constant) * (b_constant ** 2 / (4 * t_c ** 2) - 1)
print(f"Sensitivy: {sl:.2}")
G = 5 / (60 * sl)
print(f"Gain: {G:.2f}")
R_G = 50 / (G - 1)
print(f"Rg: {R_G}")

# DATA PREP
data = read_data("data_2.txt")
t_testigo = data[0]
t_medida = data[3]
v_a = data[1]
v_s = data[2]

# ------------------------------------------------
#      ----------------BRIDGE----------------
# ------------------------------------------------
print("\n\n\nBRIDGE")
t_theoretical = np.linspace(0, 100, len(t_testigo))

# Curve Fitted
param_bri, pcov_bri = curve_fit(linear, t_testigo, v_a)
err_bri = np.sqrt(np.diag(pcov_bri))

v_a_fitting = linear(t_theoretical, param_bri[0], param_bri[1])
bri_equation = f"V_sl[V] = ({param_bri[0]:.5f} ± {err_bri[0]:.2}) * t + ({param_bri[1]:.3f} ± {err_bri[1]:.2})"
print(bri_equation)

# Model
v_a_model = linear(t_theoretical, sl, 0)

# ERRORS
zero_err_bri = zero_error(param_bri)
sensitivity_err_bri = sensitivity_error(param_bri, sl)
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
plt.ylabel("$V_{sl}$ (V)")
plt.savefig("bridge_ntc")
plt.show()

# ------------------------------------------
#     ----------------AMP----------------
# ------------------------------------------
print("\n\n\nAMPLIFIER")
v_a_theoretical = np.linspace(0, sl * 100, len(v_a))

# Curve Fitted
param_amp, pcov_amp = curve_fit(linear, v_a, v_s)
err_amp = np.sqrt(np.diag(pcov_amp))

v_o_fittingV = linear(v_a_theoretical, param_amp[0], param_amp[1])
amp_equation = f"V_0[V] = ({param_amp[0]:.5f} ± {err_amp[0]:.2}) * V_sl + ({param_amp[1]:.4f} ± {err_amp[1]:.2})"
print(amp_equation)

# Model
v_o_modelV = linear(v_a_theoretical, G, 0)

# ERRORS
zero_err_amp = zero_error(param_amp)
sensitivity_err_amp = sensitivity_error(param_amp, G)
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
plt.xlabel("$V_{sl}$ (V)")
plt.ylabel("$V_0$ (V)")
plt.savefig("amplifier_ntc")
plt.show()

# ------------------------------------------------
#      ----------------SYSTEM----------------
# ------------------------------------------------
print("\n\n\nSYSTEM")
# Curve Fitted
param_sys, pcov_sys = curve_fit(linear, t_testigo, v_s)
err_sys = np.sqrt(np.diag(pcov_sys))

v_o_fittingT = linear(t_theoretical, param_sys[0], param_sys[1])
sys_equation = f"V_0[V] = ({param_sys[0]:.5f} ± {err_sys[0]:.2}) * t + ({param_sys[1]:.3f} ± {err_sys[1]:.2})"
print(sys_equation)

# Model
v_o_modelT = linear(t_theoretical, G * sl, 0)

# ERRORS
zero_err_sys = zero_error(param_sys)
sensitivity_err_sys = sensitivity_error(param_sys, 0.05)
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
plt.savefig("system_ntc")
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
plt.ylabel("T NTC ($^{\circ}$C)")
plt.savefig("temperatures_ntc")
plt.show()
# ------------------------------------------------------------- #
# -------------------------- To .txt -------------------------- #
# ------------------------------------------------------------- #

file_name = "results_ntc"
check = file_name
i = 1
while os.path.isfile(check + ".txt"):
    i += 1
    check = "%s_%d" % (file_name, i)
if i > 1:
    file_name = check

f = open(file_name + ".txt", "w")
f.writelines("\t\t\t\t\t+--------------+\n")
f.writelines("\t\t\t\t\t|  NTC SENSOR  |\n")
f.writelines("\t\t\t\t\t+--------------+\n")
f.writelines("\n\n")
f.writelines("+----------------------CHARACTERIZATION----------------------+\n")
f.writelines("  {} \n".format(ntc_equation))
f.writelines("\n")
f.writelines("\t{} \n".format("Values:"))
f.writelines(
    "\t\t{:18} {} = {:.{}} {}\n".format(
        "R design:", "\u03B5", r_design, "2f", "kΩ"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Vi max:", "\u03B5", v_i_max, "2f", "V"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Sensitivity:", "\u03B5", sl, "2f", "V/ºC"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} \n".format(
        "Gain:", "\u03B5", G, "2f"
    )
)
f.writelines(
    "\t\t{:18} {} = {:.{}} {} \n".format(
        "Rg:", "\u03B5", R_G, "2f", "kΩ"
    )
)
f.writelines("+------------------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+------------------------BRIDGE------------------------+\n")
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
f.writelines("+-------------------------------------------------------+\n")
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
f.writelines("+------------------------SYSTEM------------------------+\n")
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
f.writelines("+-------------------------------------------------------+\n")
f.writelines("\n\n")
f.writelines("+-----------------------TEMPERATURE-----------------------+\n")
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
f.writelines("+--------------------------------------------------------+\n")
f.close()
