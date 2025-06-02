# Zero-Dimensional Cardiovascular Modeling
**A Personalized Approach to Non-Invasive Measurement and Stability Analysis**

---

## Project Overview

This project applies **zero-dimensional (0D) cardiovascular modeling** to simulate and analyze blood flow dynamics in the human heart. By varying physiological parameters, the models help investigate the **personalized impact of cardiovascular traits**, with a focus on non-invasive diagnostics and system stability.

---

## Implementation Highlights

### 1. Model Development

Three core models were implemented:

- **Model 1: Single Ventricle Model**
  - A simplified lumped-parameter system combining all heart chambers into a unified pump.
  - Incorporates arterial compliance, peripheral resistance, and venous return.

- **Model 2: Four-Chamber Heart Model**
  - A detailed representation of all four cardiac chambers: RA, RV, LA, LV.
  - Each chamber modeled individually with pressure-volume characteristics and valve dynamics.

- **Model 3: Reduced Four-Chamber Model**
  - Derived from Model 2 by removing parameters identified as low-sensitivity.
  - Achieves computational efficiency with minimal accuracy loss.

---

### 2. Sensitivity Analysis

- **Techniques Used:**
  - **Morris Method** – for qualitative screening of parameter influence.
  - **Sobol Method** – for quantitative global sensitivity analysis.

- **Goal:**  
  Determine which physiological parameters most significantly impact cardiac output and pressure dynamics under normal operating ranges.

---

### 3. Parameter Rationality

- All input values were drawn from published literature to reflect **physiologically realistic** and **clinically meaningful** conditions.
- Parameter bounds were defined conservatively to ensure robustness during simulation.

---

## Results Summary

- Both models successfully replicated expected cardiovascular behaviors under varied parameter sets.
- Sensitivity analysis revealed:
  - Greater parameter influence in complex models (Model 2).
  - Simplified models (Model 1 & 3) provided reliable approximations with faster runtime.
- **Model 3** achieved a good trade-off between fidelity and performance by fixing non-critical parameters.

---

## Future Work

- Extend models by integrating:
  - **Heart rate variability**
  - **Blood viscosity**
  - **Neurohumoral feedback regulation**

- Develop alternate reduced-order models tailored for specific diagnostic or simulation purposes.

- Validate simulations with **clinical or experimental data** for real-world application.

---

## Authors
Pranav Kumar Sasikumar, Jiacheng Liu, Bhagyashree, Akio Nishida, Simran Wadhwa, Wendi Jiang  
**Supervisor:** Dr. Xu Xu

---