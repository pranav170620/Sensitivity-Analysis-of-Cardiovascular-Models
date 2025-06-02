#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 03:23:03 2024

@author: liujiacheng
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/0.008))  # Evaluation points



# Solve the differential equation
sol = solve_ivp(fun=lambda t, y: nik(t, y, params), t_span=t_span, y0=u0, t_eval=t_eval, rtol=1e-6, atol=1e-6)

# Solve for Qav and Qmv explicitly
# Qav_vals = [valve(0.033, (pLV - psa)) for pLV, psa in zip(sol.y[0], sol.y[1])]
# Qmv_vals = [valve(0.06, (psv - pLV)) for pLV, psv in zip(sol.y[0], sol.y[2])]



# Plotting
# Sub Interval of the solution
t1, t2 = 9, 10 # The time interval we focus on

indices = (sol.t >= t1) & (sol.t <= t2) 
t_sub = sol.t[indices]
subsol = []

for i in range(sol.y.shape[0]):
    subsol.append(sol.y[i, indices])

subsol = np.array(subsol)


#______________________________________________________________________________
# Global SA Sobol

start_time = time.time()

rc = 0.1 # range_concern
problem = {
    'num_vars': 9,
    'parameters': ['tau_es', 'tau_ep', 'Rmv', 'Zao', 'Rs', 'Csa', 'Csv', 'E_max', 'E_min'],
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

num_outputs = 7  # Number of outcomes
num_params = len(problem['parameters'])

def run_model(params, output_index):
    tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min = params
    updated_params = [tau_es, tau_ep, Rmv, Zao, Rs, Csa, Csv, E_max, E_min]
    
    # Explicitly defining the method and ensuring parameter types are correct
    def model_func(t, y):
        return nik(t, y, updated_params)
    
    try:
        # Running without threading to test the function
        sol = solve_ivp(model_func, t_span, u0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-6)
        output = np.var(sol.y[output_index])
        return output
    except Exception as e:
        print("Error in solve_ivp:", str(e))
        return None  # or handle appropriately

sample_sizes = [300, 500, 1000, 1500, 2500, 4000, 6000, 12000]
# Lists to store matrices for each sample size
S1_matrices = []
ST_matrices = []

def perform_analysis(N):
    param_values = sobol_sample.sample(problem, N, calc_second_order=False)
    S1_matrix = np.zeros((num_outputs, num_params))
    ST_matrix = np.zeros((num_outputs, num_params))
    
    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for params in param_values:
            for i in range(num_outputs):
                future = executor.submit(run_model, params, i)
                futures.append((future, params, i))  # Store future along with params and output index
    
    results = {}
    # Collect results as they complete
    for future, params, i in futures:
        try:
            result = future.result()
            if i not in results:
                results[i] = []
            results[i].append(result)
        except Exception as exc:
            print(f'Generated an exception: {exc}')
    
    # Now, perform Sobol analysis using the collected results
    for i in range(num_outputs):
        if i in results:
            Y = np.array(results[i])  # Convert list to NumPy array
            Si = sobol.analyze(problem, Y, calc_second_order=False)
            S1_matrix[i, :] = Si['S1']
            ST_matrix[i, :] = Si['ST']

    return S1_matrix, ST_matrix

if __name__ == '__main__':
    # Perform analysis
    for N in sample_sizes:
        S1_matrix, ST_matrix = perform_analysis(N)
        S1_matrix = np.delete(S1_matrix, [4, 5], axis=0)
        ST_matrix = np.delete(ST_matrix, [4, 5], axis=0)
        # Append results to the lists
        S1_matrices.append(S1_matrix)
        ST_matrices.append(ST_matrix)
    
    output_labels = ['pLV', 'psa', 'psv', 'Vlv', 'Qs']
    
    # Plotting the first-order sensitivity indices as a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(S1_matrix, annot=True, cmap="viridis", xticklabels=problem['parameters'], yticklabels=output_labels)
    plt.title(f'First-Order Sobol Sensitivity Indices for Each Outcome (N={N})')
    plt.xlabel('Parameter')
    plt.ylabel('Outcome')
    plt.tight_layout()
    plt.show()
    
    # Average S1 values across all outcomes for each parameter
    S1_avg = np.abs(np.mean(S1_matrix, axis=0))
    # Rank parameters by average S1 from high to low
    sorted_indices = np.argsort(S1_avg)[::-1]
    sorted_S1_avg = S1_avg[sorted_indices]
    sorted_parameters = np.array(problem['parameters'])[sorted_indices]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_parameters)), sorted_S1_avg, tick_label=sorted_parameters)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Parameter')
    plt.ylabel(f'Average Direct-Effect Index ($S1$) (N={N})')
    plt.title('Parameter Importance Ranked by Average $S1$')
    plt.tight_layout()
    plt.show()
    
    # Plotting the total-effect sensitivity indices as a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(ST_matrix, annot=True, cmap="viridis", xticklabels=problem['parameters'], yticklabels=output_labels)
    plt.title(f'Total-Effect Sobol Sensitivity Indices for Each Outcome (N={N})')
    plt.xlabel('Parameter')
    plt.ylabel('Outcome')
    plt.tight_layout()
    plt.show()
    
    # Average ST values across all outcomes for each parameter
    ST_avg = np.abs(np.mean(ST_matrix, axis=0))
    # Rank parameters by average ST from high to low
    sorted_indices = np.argsort(ST_avg)[::-1]  # Descending order
    sorted_ST_avg = ST_avg[sorted_indices]
    sorted_parameters = np.array(problem['parameters'])[sorted_indices]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_parameters)), sorted_ST_avg, tick_label=sorted_parameters)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Parameter')
    plt.ylabel('Average Total-Effect Index ($ST$)')
    plt.title(f'Parameter Importance Ranked by Average $ST$ (N={N})')
    plt.tight_layout()
    plt.show()
    
    # End time measurement
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Time taken for N={N}: {elapsed_time} seconds")
    
    # Use the elapsed time to estimate how long it might take for larger N values
    # Runs = (N*(D+2)), D = dimension space(number of parameters), N = number of samples 
    estimated_time_for_larger_N = (elapsed_time / (N * (9 + 2))) * (2048 * (9 + 2))
    print(f"Estimated time for N=2048: {estimated_time_for_larger_N} seconds")
    
    #______________________________________________________________________________
    # Parameter converging patterns
    
    num_outputs = len(output_labels) # Here it changes from 7 to 5
    num_matrices = len(ST_matrices)
    
    for j in range(num_params):
        plt.figure(figsize=(10, 6))
        # Plot ST index changes for each output with respect to the jth parameter
        for i in range(num_outputs):
            data = [ST_matrices[m][i, j] for m in range(num_matrices)]
            plt.plot(sample_sizes, data, '-o', label=output_labels[i])
    
        plt.title(f'Changes in ST Values for Parameter: {problem["parameters"][j]}')
        plt.xlabel('Sample Size [300, 500, 1000, 1500, 2500, 4000, 6000, 12000]')
        plt.ylabel('ST Index')
        plt.xticks(sample_sizes)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
