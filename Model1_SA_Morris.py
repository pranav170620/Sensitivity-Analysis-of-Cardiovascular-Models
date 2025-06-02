#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 02:41:20 2024

@author: liujiacheng
"""

from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from SALib.test_functions import Sobol_G
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# Setting for the equations
def valve(R, deltaP):
    if -deltaP < 0:
        q = deltaP / R
    else:
        q = 0
    return q

def shi_elastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift):
    ti = (t + (1 - Eshift) * tau) % tau 
    Ep = 0
    if ti <= tau_es:
        Ep = (1 - np.cos(ti / tau_es * np.pi)) / 2
    elif tau_es < ti <= tau_ep:
        Ep = (1 + np.cos((ti - tau_es) / (tau_ep - tau_es) * np.pi)) / 2
    E = E_min + (E_max - E_min) * Ep
    return E

def d_shi_elastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift):
    ti = (t + (1 - Eshift) * tau) % tau
    DEp = 0
    if ti <= tau_es:
        DEp = np.pi / tau_es * np.sin(ti / tau_es * np.pi) / 2
    elif tau_es < ti <= tau_ep:
        DEp = np.pi / (tau_ep - tau_es) * np.sin((tau_es - ti) / (tau_ep - tau_es) * np.pi) / 2
    DE = (E_max - E_min) * DEp
    return DE

# Setting of ODE
def nik(t, u, p):
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = p
    du = np.zeros_like(u)

    E = shi_elastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift)
    dE = d_shi_elastance(t, E_min, E_max, tau, tau_es, tau_ep, Eshift)
    
    Qav = valve(Zao, (pLV - psa))
    Qmv = valve(Rmv, (psv - pLV))
    
    du[0] = (Qmv - Qav) * E + pLV / E * dE
    du[1] = (Qav - Qs) / Csa
    du[2] = (Qs - Qmv) / Csv
    du[3] = Qmv - Qav
    du[4] = 0
    du[5] = 0
    du[6] = (du[1] - du[2]) / Rs
    
    return du



# Initial conditions and parameters
Eshift = 0.0
E_min = 0.03
tau_es = 0.3
tau_ep = 0.45
E_max = 1.5
Rmv = 0.06
Zao = 0.033
Rs = 1.11
Csa = 1.13
Csv = 11.0

tau = 1 # Set to 1 for simplication

u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]
params = [tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
# Define the time span of the integration
t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/0.002))  # Evaluation points



# Solve the differential equation
sol = solve_ivp(fun=lambda t, y: nik(t, y, params), t_span=t_span, y0=u0, t_eval=t_eval, rtol=1e-6, atol=1e-6)

# Solve for Qav and Qmv explicitly
# Qav_vals = [valve(0.033, (pLV - psa)) for pLV, psa in zip(sol.y[0], sol.y[1])]
# Qmv_vals = [valve(0.06, (psv - pLV)) for pLV, psv in zip(sol.y[0], sol.y[2])]



# Plotting
# Sub Interval of the solution
t1, t2 = 9, 10 # The time interval we focus on

indices = (sol.t >= t1-0.05) & (sol.t <= t2) 
t_sub = sol.t[indices]
subsol = []

for i in range(sol.y.shape[0]):
    subsol.append(sol.y[i, indices])

subsol = np.array(subsol)



#______________________________________________________________________________
# SA Morris

start_time = time.time()

rc = 0.1 # range_concern
problem = {
    'num_vars': 9,
    'names': ['tau_es', 'tau_ep', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
#    'bounds': [
#        [0.21, 0.34],  # tau_es bounds
#        [0.36, 0.585],  # tau_ep bounds
#        [0.042, 0.078],  # Rmv bounds
#        [0.0231, 0.0429],  # Zao bounds
#        [0.777, 1.443],  # Rs bounds
#        [0.791, 1.469],  # Csa bounds
#        [7.7, 14.3],  # Csv bounds
#        [1.05, 1.95],  # E_max bounds
#        [0.021, 0.039]  # E_min bounds
#    ]
    'bounds': [
        [tau_es*(1-rc), tau_es*(1+rc)],
        [tau_ep*(1-rc), tau_ep*(1+rc)],
        [Rmv*(1-rc), Rmv*(1+rc)],
        [Zao*(1-rc), Zao*(1+rc)],
        [Rs*(1-rc), Rs*(1+rc)],
        [Csa*(1-rc), Csa*(1+rc)],
        [Csv*(1-rc), Csv*(1+rc)],
        [E_max*(1-rc), E_max*(1+rc)],
        [E_min*(1-rc), E_min*(1+rc)]
    ]
}

N = 2000  # Number of samples
param_values = morris.sample(problem, N, num_levels=10) # Change num_levels too

def run_model(params):
    sol = solve_ivp(fun=lambda t, y: nik(t, y, params), t_span=t_span, y0=u0, t_eval=t_eval, rtol=1e-6, atol=1e-6)
    return np.var(sol.y, axis=1)

# Run the model for each parameter set
Y = np.array([run_model(params) for params in param_values])

# Perform Morris analysis for each output variable
morris_results = [morris_analyze.analyze(problem, param_values, Y[:, i], print_to_console=False) for i in range(Y.shape[1])]

mu_matrix = np.zeros((len(morris_results), problem['num_vars']))
sigma_matrix = np.zeros((len(morris_results), problem['num_vars']))

for i, result in enumerate(morris_results):
    mu_matrix[i, :] = result['mu']
    sigma_matrix[i, :] = result['sigma']

# Labels
output_labels = ['pLV', 'psa', 'psv', 'Vlv', 'Qs']
parameter_names = problem['names']

# Normalize the mu_matrix values to range [-1, 1], Initially (-Inf, Inf)
max_abs_value = np.max(np.abs(mu_matrix))
mu_matrix_normalized = mu_matrix / max_abs_value
# Normalize the sigma_matrix values to range [0, 1], Initially [0, Inf)
max_abs_value = np.max(np.abs(sigma_matrix))
sigma_matrix_normalized = sigma_matrix / max_abs_value

mu_matrix_normalized = np.delete(mu_matrix_normalized, [4, 5], axis=0)
sigma_matrix_normalized = np.delete(sigma_matrix_normalized, [4, 5], axis=0)

# Plotting the normalized mu_matrix
plt.figure(figsize=(12, 6))
sns.heatmap(mu_matrix_normalized, annot=True, cmap="viridis", xticklabels=parameter_names, yticklabels=output_labels, vmin=-1, vmax=1)
plt.title(f'Normalized Mean of the Elementary Effects (N={N})')
plt.xlabel('Parameter')
plt.ylabel('Outcome')
plt.tight_layout()
plt.show()

# Average values across all outcomes for each parameter
mu_matrix_avg = np.abs(np.mean(mu_matrix_normalized, axis=0))
# Rank parameters from high to low
sorted_indices = np.argsort(mu_matrix_avg)[::-1]  # Descending order
sorted_avg = mu_matrix_avg[sorted_indices]
sorted_parameters = np.array(problem['names'])[sorted_indices]
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_parameters)), sorted_avg, tick_label=sorted_parameters)
plt.xticks(rotation=45, ha="right")
plt.xlabel('Parameter')
plt.ylabel('Average Outcome Index (Mean Elementary Effect)')
plt.title(f'Parameter Importance (N={N})')
plt.tight_layout()
plt.show()

# Plotting the normalized sigma_matrix
plt.figure(figsize=(12, 6))
sns.heatmap(sigma_matrix_normalized, annot=True, cmap="viridis", xticklabels=parameter_names, yticklabels=output_labels)
plt.title(f'Normalized Standard Deviation of the Elementary Effects (N={N})')
plt.xlabel('Parameter')
plt.ylabel('Outcome')
plt.tight_layout()
plt.show()

# Timing
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken for N={N}: {elapsed_time} seconds")

















