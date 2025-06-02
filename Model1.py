#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:11:47 2024

@author: liujiacheng

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
Qav_vals = [valve(0.033, (pLV - psa)) for pLV, psa in zip(sol.y[0], sol.y[1])]
Qmv_vals = [valve(0.06, (psv - pLV)) for pLV, psv in zip(sol.y[0], sol.y[2])]



# Plotting
# Sub Interval of the solution
t1, t2 = 8, 10 # The time interval we focus on

indices = (sol.t >= t1) & (sol.t <= t2) 
t_sub = sol.t[indices]
subsol = []

for i in range(sol.y.shape[0]):
    subsol.append(sol.y[i, indices])

subsol = np.array(subsol)

# Plot of Pressure over time
plt.figure(figsize=(10, 6))
for i, label in enumerate(["P_LV", "P_SA"]):
    plt.plot(sol.t, sol.y[i], label=label)
plt.legend(fontsize='large')
plt.title("Pressure over time", fontsize = 20)
plt.xlabel("Time (s)", fontsize = 16)
plt.ylabel("Pressure (mmHg)", fontsize = 16)
plt.xlim(t1, t2)
plt.grid(True) 
plt.show()

# Plot of PV-Loop
plt.figure(figsize=(10, 6))
plt.plot(subsol[3], subsol[0])
plt.title("PV-Loop", fontsize = 20)
plt.xlabel("Volume (mL)", fontsize = 16)
plt.ylabel("Pressure (mmHg)", fontsize = 16)
plt.grid(True)
plt.show()

# Plot of Volume over time
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[3], label="V_LV")  # Adjust the index for V_LV
plt.legend(fontsize='large')
plt.title("Volume over time", fontsize = 20)
plt.xlabel("Time (s)", fontsize = 16)
plt.ylabel("Volume (mL)", fontsize = 16)
plt.xlim(t1, t2)
plt.ylim(80,170)
plt.grid(True)
plt.show()

#Plot of in-flow rate
plt.figure(figsize=(12, 6))
plt.plot(sol.t, Qav_vals, label='Qav')
plt.plot(sol.t, Qmv_vals, label='Qmv')
plt.legend()
plt.title("Flow Rates Over Time")
plt.xlabel("Time")
plt.ylabel("Flow Rate")
plt.xlim(9, 10)
plt.show()

