#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 01:32:22 2024

@author: liujiacheng
"""

# To better explain each of the parameters, I will include an explanation of each with the value selected:
    
"""
Initial Conditions

Vsa (623 mL): Volume of Systemic Arteries
Vsv (3115 mL): Volume of Systemic Veins
Vpa (144 mL): Volume of Pulmonary Arteries
Vpv (527 mL): Volume of Pulmonary Veins
VraM (58.2 mL): Maximum Volume of Right Atrium
VlaM (58.2 mL): Maximum Volume of Left Atrium
VrvM (151.3 mL): Maximum Volume of Right Ventricle
VlvM (151.3 mL): Maximum Volume of Left Ventricle

Parameters

Resistances:
Rs (0.86): Systemic Resistance - The resistance to blood flow in the systemic circulation
Rp (0.036): Pulmonary Resistance - The resistance to blood flow in the pulmonary circulation
Rava (0.014): Aortic Valve Resistance
Rmva (0.003): Mitral Valve Resistance
Rpva (0.0025): Pulmonary Valve Resistance
Rtva (0.003): Tricuspid Valve Resistance
Rpv (0.036): Pulmonary Veins Resistance - The resistance to blood flow in the pulmonary veins
Rsv (0.072): Systemic Veins Resistance - The resistance to blood flow in the systemic veins

Compliances:
Csa (1.8): Compliance of Systemic Arteries - The ability of systemic arteries to stretch
Csv (31.2): Compliance of Systemic Veins - The ability of systemic veins to stretch
Cpa (7.0): Compliance of Pulmonary Arteries
Cpv (11.6): Compliance of Pulmonary Veins

Elastance (Ability of the heart chambers to contract and expand):
EMra (0.79): Max Elastance of Right Atrium - The maximum ability of the right atrium to contract
Emra (0.04): Min Elastance of Right Atrium - The minimum ability of the right atrium to contract
EMla (0.85): Max Elastance of Left Atrium
Emla (0.04): Min Elastance of Left Atrium
EMrv (0.34): Max Elastance of Right Ventricle 
Emrv (0.021): Min Elastance of Right Ventricle
EMlv (1.92): Max Elastance of Left Ventricle
Emlv (0.055): Min Elastance of Left Ventricle

Timing (Phases of cardiac cycle)
Trra (0.1): Time Right Atrium Relaxation Ends
tcra (0.8): Time Right Atrium Contraction Begins 
Tcra (0.97): Time Right Atrium Contraction Ends
Tcrv (0.3): Time Right Ventricle Contraction Begins 
Trrv (0.51): Time Right Ventricle Relaxation Begins 

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Global values for recording
pressures = {'psa': [], 'psv': [], 'ppa': [], 'ppv': [], 'pla': [], 'pra': [], 'plv': [], 'prv': [], 't': []}
    
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
    
    # Need to record pressure data as well but not as outputs
    global pressures
    
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
    
    pressures['t'].append(t)
    pressures['psa'].append(psa)
    pressures['psv'].append(psv)
    pressures['ppa'].append(ppa)
    pressures['ppv'].append(ppv)
    pressures['pla'].append(pla)
    pressures['pra'].append(pra)
    pressures['plv'].append(plv)
    pressures['prv'].append(prv)
    
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

"""

Important

All of the initial values here are provided indirectly in load_global_CTR.m file and requires some calculation
Need careful calculation with height = 180, weight = 75, gender = 2
However we could also try with other reasonable values

I have checked that all of the values below fall within the suitable ranges
Yet it would be better to do some formal research of the ranges and state that the initial values below are reasonable

"""

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

for key in pressures:
    pressures[key] = np.array(pressures[key])

# Ensure alignment (optional, if necessary, interpolate or extract matching times)
# Example: Extract pressure values at the exact times returned in result.t
pla = np.interp(result.t, pressures['t'], pressures['pla'])
pra = np.interp(result.t, pressures['t'], pressures['pra'])
plv = np.interp(result.t, pressures['t'], pressures['plv'])
prv = np.interp(result.t, pressures['t'], pressures['prv'])


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
pla_i = np.interp(t_interval, pressures['t'], pressures['pla'])
pra_i = np.interp(t_interval, pressures['t'], pressures['pra'])
plv_i = np.interp(t_interval, pressures['t'], pressures['plv'])
prv_i = np.interp(t_interval, pressures['t'], pressures['prv'])
psa_i = np.interp(t_interval, pressures['t'], pressures['psa'])
psv_i = np.interp(t_interval, pressures['t'], pressures['psv'])
ppa_i = np.interp(t_interval, pressures['t'], pressures['ppa'])
ppv_i = np.interp(t_interval, pressures['t'], pressures['ppv'])



# Plotting the PV-loops for the ventricles

# Ventricle PV-loops
plt.figure(figsize=(10, 6))
plt.plot(Vlv_i, plv_i, label='Left Ventricle')
plt.plot(Vrv_i, prv_i, label='Right Ventricle')
plt.title('Ventricular Pressure-Volume Loops')
plt.xlabel('Volume (mL)')
plt.ylabel('Pressure (mmHg)')
plt.grid(True) 
plt.legend()

# Atrial PV-loops
plt.figure(figsize=(10, 6))
plt.plot(Vla_i, pla_i, label='Left Atrium')
plt.plot(Vra_i, pra_i, label='Right Atrium')
plt.title('Atrial Pressure-Volume Loops')
plt.xlabel('Volume (mL)')
plt.ylabel('Pressure (mmHg)')
plt.grid(True) 
plt.legend()

plt.tight_layout()
plt.show()

# Model Outputs: Volumns
plt.figure(figsize=(14, 7))

# Systemic Arterial Volume
plt.subplot(1, 2, 1)
plt.plot(t_interval, Vsa_i, label='Vsa (Systemic Arterial Volume)', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Volume (mL)')
plt.title('Systemic Arterial Volume over Time')
plt.grid(True)
plt.legend()

# Systemic Venous Volume
plt.subplot(1, 2, 2)
plt.plot(t_interval, Vsv_i, label='Vsv (Systemic Venous Volume)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Volume (mL)')
plt.title('Systemic Venous Volume over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
