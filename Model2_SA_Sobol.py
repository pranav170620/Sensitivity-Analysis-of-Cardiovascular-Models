#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 04:17:40 2024

@author: liujiacheng
"""

from concurrent.futures import ProcessPoolExecutor
from SALib.sample import sobol as sobol_sample
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import time
 
def elastance_atrium(t, EaM, Eam, Tar, tac, Tac, T):
    ti = t % T
    if ti <= Tar:
        return (EaM - Eam) * (1 - np.cos(np.pi * (ti - Tar) / (T - Tac + Tar))) / 2 + Eam
    elif Tar < ti <= tac:
        return Eam
    elif tac < ti <= Tac:
        return (EaM - Eam) * (1 - np.cos(np.pi * (ti - tac) / (Tac - tac))) / 2 + Eam
    else:
        return (EaM - Eam) * (1 + np.cos(np.pi * (ti - Tac) / (T - Tac + Tar))) / 2 + Eam

def elastance_ventricle(t, EvM, Evm, Tvc, Tvr, T):
    ti = t % T
    if ti <= Tvc:
        return (EvM - Evm) * (1 - np.cos(np.pi * ti / Tvc)) / 2 + Evm
    elif t <= Tvr:
        return (EvM - Evm) * (1 + np.cos(np.pi * (ti - Tvc) / (Tvr - Tvc))) / 2 + Evm
    else:
        return Evm
    
def model_basic(t, y, pars, Vunstr, T):
    
    # Unpack the state variables for clarity
    Vsa, Vsv, Vpa, Vpv, Vra, Vla, Vrv, Vlv = y
    
    # Unpack the unstressed volumes
    VsaU, VsvU, VpaU, VpvU, VaU, VvU = Vunstr
    
    # Unpack parameters
    Rs, Rp, Rava, Rmva, Rpva, Rtva, Rpv, Rsv = pars[:8]
    Csa, Csv, Cpa, Cpv = pars[8:12]
    EMra, Emra, EMla, Emla, EMrv, Emrv, EMlv, Emlv = pars[12:20]
    Trra, tcra, Tcra, Tcrv, Trrv = pars[20:25]
    
    # Calculate elastance using previously defined functions
    Era = elastance_atrium(t, EMra, Emra, Trra, tcra, Tcra, T)
    Ela = elastance_atrium(t, EMla, Emla, Trra, tcra, Tcra, T)
    Erv = elastance_ventricle(t, EMrv, Emrv, Tcrv, Trrv, T)
    Elv = elastance_ventricle(t, EMlv, Emlv, Tcrv, Trrv, T)
    
    # Compute pressures in each compartment
    psa = (Vsa - VsaU) / Csa
    psv = (Vsv - VsvU) / Csv
    ppa = (Vpa - VpaU) / Cpa
    ppv = (Vpv - VpvU) / Cpv
    
    # Placeholder for pressure computation using elastance
    pla = Ela * (Vla - VaU)
    pra = Era * (Vra - VaU)
    plv = Elv * (Vlv - VvU)
    prv = Erv * (Vrv - VvU)
    
    # Compute flows (based on pressures and resistances
    qava = max((plv - psa) / Rava, 0) if plv > psa else 0
    qmva = max((pla - plv) / Rmva, 0) if pla > plv else 0
    qpva = max((prv - ppa) / Rpva, 0) if prv > ppa else 0
    qtva = max((pra - prv) / Rtva, 0) if pra > prv else 0
    qsv = (psv - pra) / Rsv
    
    # Differential equations for volumes
    dVsa = qava - (psa - psv) / Rs
    dVsv = (psa - psv) / Rs - qsv
    dVpa = qpva - (ppa - ppv) / Rp
    dVpv = (ppa - ppv) / Rp - (ppv - pla) / Rpv
    dVra = qsv - qtva
    dVla = (ppv - pla) / Rpv - qmva
    dVrv = qtva - qpva
    dVlv = qmva - qava

    return([dVsa, dVsv, dVpa, dVpv, dVra, dVla, dVrv, dVlv])  # Volumes

def unstressed_volumes(Vsa, Vsv, Vpa, Vpv, Va, Vv):
    VsaU = Vsa * 0.73  # 27% unstressed
    VsvU = Vsv * 0.92  # 8% unstressed
    VpaU = Vpa * 0.42  # 58% unstressed
    VpvU = Vpv * 0.89  # 11% unstressed
    VaU  = Va
    VvU  = Vv
    return [VsaU, VsvU, VpaU, VpvU, VaU, VvU]

def simulate_model(pars, initial_conditions, Vunstr, T, t_span, a):
    solution = solve_ivp(lambda t, y: model_basic(t, y, pars, Vunstr, T), t_span, initial_conditions, rtol = a, atol = a)
    return solution

initial_conditions = [623, 3115, 144, 527, 58.2, 58.2, 151.3, 151.3]   # Vsa, Vsv, Vpa, Vpv, VraM, VlaM, VrvM, VlvM

parameters = [
    0.86, 0.036, 0.014, 0.003, 0.0025, 0.003, 0.036, 0.072,  # Resistances Rs, Rp, Rava, Rmva, Rpva, Rtva, Rpv, Rsv
    1.8, 31.2, 7.0, 11.6,  # Compliances Csa, Csv, Cpa, Cpv
    0.79, 0.04, 0.85, 0.04, 0.34, 0.021, 1.92, 0.055,  # Elastance EMra, Emra, EMla, Emla, EMrv, Emrv, EMlv, Emlv
    0.1, 0.8, 0.97, 0.3, 0.51  # Timing Trra, tcra, Tcra, Tcrv, Trrv
]

Vsa, Vsv, Vpa, Vpv, Va, Vv = 623, 3115, 144, 527, 10, 5
Vunstr = unstressed_volumes(Vsa, Vsv, Vpa, Vpv, Va, Vv)

T = 1.0 # This is the time period for 1 cycle
t_span = (0, 10 * T)

result = simulate_model(parameters, initial_conditions, Vunstr, T, t_span, 1e-10)

# Subinterval to plot
start_time = 4 * T
end_time = 10 * T

# Filter data for specific time interval
interval_indices = (result.t >= start_time) & (result.t <= end_time)

# Extract data for the interval
t_interval = result.t[interval_indices]
Vsa_i = result.y[0, interval_indices]
Vsv_i = result.y[1, interval_indices]
Vpa_i = result.y[2, interval_indices]
Vpv_i = result.y[3, interval_indices]
Vra_i = result.y[4, interval_indices]
Vla_i = result.y[5, interval_indices]
Vrv_i = result.y[6, interval_indices]
Vlv_i = result.y[7, interval_indices]


#______________________________________________________________________________
# Global SA Sobol

def run_model(params, output_index):
    global pressures
    pressures = {'psa': [], 'psv': [], 'ppa': [], 'ppv': [], 'pla': [], 'pra': [], 'plv': [], 'prv': [], 't': []}
    # Unpack all parameters. Ensure the order of params matches the order used in sampling.
    Rs, Rp, Rava, Rmva, Rpva, Rtva, Rpv, Rsv, Csa, Csv, Cpa, Cpv, \
    EMra, Emra, EMla, Emla, EMrv, Emrv, EMlv, Emlv, Trra, tcra, Tcra, Tcrv, Trrv = params
    
    # Prepare parameters for the model, can be a dictionary or directly as arguments
    model_params = [
        Rs, Rp, Rava, Rmva, Rpva, Rtva, Rpv, Rsv, 
        Csa, Csv, Cpa, Cpv, 
        EMra, Emra, EMla, Emla, EMrv, Emrv, EMlv, Emlv, 
        Trra, tcra, Tcra, Tcrv, Trrv
    ]

    T = 1.0
    t_span = (0, 10 * T)
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/0.008))  # Evaluation points
    # Solve the ODE with the parameters
    sol = solve_ivp(fun=lambda t, y: model_basic(t, y, model_params, Vunstr, T), t_span=t_span, y0=initial_conditions, t_eval=t_eval, rtol=1e-10, atol=1e-10)

    # Compute the variance of the output
    output = np.mean(sol.y[output_index])

    return output

start_time = time.time()

rc = 0.01  # range_concern
initial_values = {
    'Rs': 0.86, 'Rp': 0.036, 'Rava': 0.014, 'Rmva': 0.003, 'Rpva': 0.0025, 'Rtva': 0.003,
    'Rpv': 0.036, 'Rsv': 0.072, 'Csa': 1.8, 'Csv': 31.2, 'Cpa': 7.0, 'Cpv': 11.6,
    'EMra': 0.79, 'Emra': 0.04, 'EMla': 0.85, 'Emla': 0.04, 'EMrv': 0.34, 'Emrv': 0.021,
    'EMlv': 1.92, 'Emlv': 0.055, 'Trra': 0.1, 'tcra': 0.8, 'Tcra': 0.97, 'Tcrv': 0.3, 'Trrv': 0.51
}

param_names = list(initial_values.keys())
bounds = [[value * (1 - rc), value * (1 + rc)] for value in initial_values.values()]

problem = {
    'num_vars': len(param_names),
    'parameters': param_names,
    'bounds': bounds
}

# Initialize matrices to hold sensitivity indices
num_outputs = 8  # Number of outcomes
num_params = len(problem['parameters'])  # Number of parameters

sample_sizes = [12000]
# Lists to store matrices for each sample size
S1_matrices = []
ST_matrices = []

def perform_analysis(N):
    param_values = sobol_sample.sample(problem, N, calc_second_order=False)
    S1_matrix = np.zeros((num_outputs, num_params))
    ST_matrix = np.zeros((num_outputs, num_params))
    
    futures = []
    with ProcessPoolExecutor(max_workers=8) as executor:
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
# Parameter patterns

# num_outputs = len(output_labels)
# num_matrices = len(ST_matrices)

# for j in range(num_params):
#    plt.figure(figsize=(10, 6))
#    for i in range(num_outputs):
#        # Extract the specific parameter index across all matrices for the ith output
#        data = [ST_matrices[m][i, j] for m in range(num_matrices)]
#        plt.plot(range(1, num_matrices + 1), data, '-o', label=output_labels[i])

#    plt.title(f'Changes in ST Values for Parameter: {problem["parameters"][j]}')
#    plt.xlabel('Intermediate Step in Inverse Square Root Distribution')
#    plt.ylabel('ST Index')
#    plt.xticks(range(1, num_matrices + 1))
#    plt.legend()
#    plt.grid(True)
#    plt.show()











