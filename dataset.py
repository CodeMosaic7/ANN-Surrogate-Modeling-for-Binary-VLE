import numpy as np
import pandas as pd

def Psat(T, A, B, C):
    """Antoine Equation: returns vapor pressure (mmHg) at T in °C"""
    return 10 ** (A - B / (T + C))

antoine = {
    "ethanol": (8.20417, 1642.89, 230.3),
    "water":   (8.07131, 1730.63, 233.426)
}

# Margules 1-parameter model
def gamma_Margules(x1, A12=1.6):
    """
    Returns activity coefficients (γ1, γ2) for binary system
    using 1-parameter Margules model.
    A12 controls strength of non-ideality (higher → stronger azeotropy).
    """
    x2 = 1 - x1
    ln_gamma1 = A12 * (x2**2)
    ln_gamma2 = A12 * (x1**2)
    return np.exp(ln_gamma1), np.exp(ln_gamma2)

# Dataset generation
x1_vals = np.linspace(0.01, 0.99, 100)   # liquid mole fraction ethanol
T_vals = np.linspace(60, 100, 100)       # °C
P = 760  # mmHg = 1 atm

rows = []
for x1 in x1_vals:
    for T in T_vals:
        # Vapor pressures
        P1_sat = Psat(T, *antoine["ethanol"])
        P2_sat = Psat(T, *antoine["water"])

        # Activity coefficients (Margules)
        g1, g2 = gamma_Margules(x1, A12=1.6)

        # Modified Raoult's Law with γ
        num = x1 * g1 * P1_sat
        den = num + (1 - x1) * g2 * P2_sat
        y1 = num / den

        rows.append([x1, T, P, y1])

data = pd.DataFrame(rows, columns=["x1", "T", "P", "y1"])
data.to_csv("vle_data_margules.csv", index=False)
print("Dataset saved as vle_data_margules.csv with", len(data), "points")


