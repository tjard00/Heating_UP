"""
Chamber model with:
- closed outlet until total pressure reaches p_set
- then "ideal" venting to keep total pressure at p_set (vent to lower backpressure)
- humid air initially (given RH)
- steam inflow (quality x=1, assumed saturated vapor at T_steam_in_C)
- constant CO2 desorption mass source with additional heat sink Q_des (J/kg_CO2)
- water vapor can condense to liquid to satisfy saturation: p_H2O <= p_sat(T)

Outputs (time series):
- total pressure and partial pressures (air, H2O, CO2)
- temperature
- mass flow rates: in (steam, CO2), condensed, out (total + components)

ASSUMPTIONS (kept simple and explicit):
- ideal gases, perfectly mixed gas phase
- one lumped temperature for the whole chamber (dominated by cp-mass heat capacity)
- no external heat loss to ambient (can be added as Q_dot_loss if needed)
- steam inflow energy uses saturated-vapor enthalpy at T_steam_in_C (approx correlations)
- condensation releases latent heat h_fg(T) (approx correlation)
- vented gas removes sensible enthalpy only (approx constant cp per species)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0) Constants
# -----------------------------
R = 8.314462618  # J/mol/K

M_H2O = 0.01801528  # kg/mol
M_CO2 = 0.0440095   # kg/mol
M_air = 0.0289652   # kg/mol (mean molar mass dry air)

# Rough (constant) molar heat capacities for sensible enthalpy of vented gases (J/mol/K)
cp_m_air = 29.1
cp_m_CO2 = 37.1
cp_m_H2O = 33.6  # water vapor

T_ref_K = 273.15  # reference for sensible enthalpy (0°C)

# -----------------------------
# 1) User parameters (your case)
# -----------------------------
V_ch = 2.7  # m^3

m_cp = 2010.0  # kg
cp_cp = 850.0  # J/kg/K
C_cp = m_cp * cp_cp  # J/K (dominant thermal mass)

mdot_steam_in = 0.1212  # kg/s
T_steam_in_C = 85.0     # °C
x_steam = 1.0           # quality (1 = dry saturated vapor)

mdot_CO2_des = 0.0147   # kg/s
Q_des_J_per_kgCO2 = 1770e3  # J/kg_CO2 (extra heat sink)

dt = 1.0  # s
t_end = 700.0  # s (adjust as you like)

# Initial conditions
p1_mbar = 100.0
p1 = p1_mbar * 100.0  # Pa
T1_C = 10.0
T1 = T1_C + 273.15  # K
RH1 = 0.05          # 5% relative humidity (0..1)

# Pressure control
p_set_mbar = 580.0
p_set = p_set_mbar * 100.0  # Pa

# External heat input (if any)
Qdot_ext = 0.0  # W (set if you have external heating)

# -----------------------------
# 2) Thermo helper functions
# -----------------------------
def p_sat_water_Pa(T_K: float) -> float:
    """
    Saturation vapor pressure of water (Pa).
    Magnus-Tetens (good for ~0..100°C).
    """
    T_C = T_K - 273.15
    # over liquid water:
    return 610.94 * np.exp((17.625 * T_C) / (T_C + 243.04))

def h_fg_water_J_per_kg(T_K: float) -> float:
    """
    Approx latent heat of vaporization of water (J/kg) for ~0..100°C.
    Linear-ish correlation: h_fg(kJ/kg) ~ 2500.9 - 2.381*T(°C)
    """
    T_C = T_K - 273.15
    return (2500.9 - 2.381 * T_C) * 1000.0

def h_f_liquid_J_per_kg(T_K: float) -> float:
    """
    Approx liquid water enthalpy relative to 0°C (J/kg).
    h_f ~ cp_l * (T - 0°C). cp_l ~ 4.186 kJ/kg/K
    """
    cp_l = 4186.0  # J/kg/K
    return cp_l * (T_K - 273.15)

def h_g_sat_J_per_kg(T_K: float) -> float:
    """
    Approx saturated vapor enthalpy (J/kg) at T, relative to 0°C baseline:
    h_g = h_f + h_fg.
    """
    return h_f_liquid_J_per_kg(T_K) + h_fg_water_J_per_kg(T_K)

def molar_enthalpies_J_per_mol(T_K: float):
    """
    Species enthalpies relative to T_ref_K (J/mol).
    Water uses saturated vapor enthalpy (includes latent heat).
    """
    dT = T_K - T_ref_K
    h_air = cp_m_air * dT
    h_co2 = cp_m_CO2 * dT
    h_h2o = h_g_sat_J_per_kg(T_K) * M_H2O
    return h_air, h_h2o, h_co2

T_steam_in_K = T_steam_in_C + 273.15
h_in_steam = x_steam * h_g_sat_J_per_kg(T_steam_in_K) + (1.0 - x_steam) * h_f_liquid_J_per_kg(T_steam_in_K)
qdot_steam_in_const = mdot_steam_in * h_in_steam  # W
qdot_des_const = mdot_CO2_des * Q_des_J_per_kgCO2  # W
p_sat_in_Pa = p_sat_water_Pa(T_steam_in_K)

# -----------------------------
# 3) Initialize state from RH
# -----------------------------
p_sat1 = p_sat_water_Pa(T1)
p_H2O_1 = RH1 * p_sat1
p_air_1 = max(p1 - p_H2O_1, 0.0)

n_air = p_air_1 * V_ch / (R * T1)
n_h2o = p_H2O_1 * V_ch / (R * T1)
n_co2 = 0.0

m_cond_liq = 0.0  # kg accumulated liquid water (condensate)

# -----------------------------
# 4) Allocate time series
# -----------------------------
N = int(np.floor(t_end / dt)) + 1
t = np.arange(N) * dt

T = np.zeros(N)
p_tot = np.zeros(N)
p_air = np.zeros(N)
p_h2o = np.zeros(N)
p_co2 = np.zeros(N)
p_sat_chamber = np.zeros(N)
phi_rel = np.zeros(N)

# mass flow signals (kg/s)
mdot_in_steam = np.full(N, mdot_steam_in)
mdot_in_co2 = np.full(N, mdot_CO2_des)
mdot_cond = np.zeros(N)

mdot_out_tot = np.zeros(N)
mdot_out_air = np.zeros(N)
mdot_out_h2o = np.zeros(N)
mdot_out_steam = np.zeros(N)
mdot_out_co2 = np.zeros(N)

qdot_steam_in = np.full(N, qdot_steam_in_const)
qdot_des = np.full(N, qdot_des_const)
qdot_ext_series = np.full(N, Qdot_ext)
qdot_cond = np.zeros(N)
qdot_out = np.zeros(N)
qdot_heating_chamber = np.zeros(N)
qdot_balance_res = np.zeros(N)

# Phase flag
venting_active = False

# -----------------------------
# 5) Time stepping
# -----------------------------
T_prev = T1

for k in range(N):
    # Record current state BEFORE applying step (k=0 is initial)
    n_tot_now = n_air + n_h2o + n_co2
    p_now = n_tot_now * R * T_prev / V_ch
    p_sat_now = p_sat_water_Pa(T_prev)
    p_air_now = n_air * R * T_prev / V_ch
    p_h2o_now = n_h2o * R * T_prev / V_ch
    p_co2_now = n_co2 * R * T_prev / V_ch

    T[k] = T_prev
    p_tot[k] = p_now
    p_air[k] = p_air_now
    p_h2o[k] = p_h2o_now
    p_co2[k] = p_co2_now
    p_sat_chamber[k] = p_sat_now
    phi_rel[k] = p_h2o_now / max(p_sat_now, 1e-9)

    if k == N - 1:
        break

    # Activate venting once we reach set pressure
    p_limit_now = max(p_set, p_sat_now)
    if (not venting_active) and (p_now >= p_limit_now):
        venting_active = True

    # ---- Sources this step (mol added)
    dn_h2o_in = (mdot_steam_in * dt) / M_H2O
    dn_co2_in = (mdot_CO2_des * dt) / M_CO2

    n_air_star = n_air
    n_h2o_star = n_h2o + dn_h2o_in
    n_co2_star = n_co2 + dn_co2_in

    # ---- Energy terms that do not depend on T_guess too much
    dH_in = qdot_steam_in_const * dt  # J
    dQ_des = qdot_des_const * dt  # J

    # ---- Solve for T_next with fixed-point iteration (handles condensation + venting coupling)
    T_guess = T_prev
    for _ in range(80):
        # Condensation equilibrium at T_guess
        p_sat = p_sat_water_Pa(T_guess)
        n_h2o_eq = p_sat * V_ch / (R * T_guess)
        n_h2o_gas = min(n_h2o_star, n_h2o_eq)

        dm_cond_step = max(0.0, (n_h2o_star - n_h2o_gas) * M_H2O)  # kg in this step
        dH_cond = dm_cond_step * h_fg_water_J_per_kg(T_guess)       # J released

        # Pre-vent moles after condensation
        n_air_pre = n_air_star
        n_h2o_pre = n_h2o_gas
        n_co2_pre = n_co2_star
        n_tot_pre = n_air_pre + n_h2o_pre + n_co2_pre

        # Venting required to meet set pressure (if active)
        if venting_active:
            p_limit_iter = max(p_set, p_sat)
            n_tot_set = p_limit_iter * V_ch / (R * T_guess)
            n_out = max(0.0, n_tot_pre - n_tot_set)
        else:
            n_out = 0.0

        if n_tot_pre > 0.0:
            y_air = n_air_pre / n_tot_pre
            y_h2o = n_h2o_pre / n_tot_pre
            y_co2 = n_co2_pre / n_tot_pre
        else:
            y_air = y_h2o = y_co2 = 0.0

        if n_tot_pre > 0.0:
            n_air_out_iter = y_air * n_out
            n_h2o_out_iter = y_h2o * n_out
            n_co2_out_iter = y_co2 * n_out
            h_air_mol, h_h2o_mol, h_co2_mol = molar_enthalpies_J_per_mol(T_guess)
            dH_out = (
                n_air_out_iter * h_air_mol
                + n_h2o_out_iter * h_h2o_mol
                + n_co2_out_iter * h_co2_mol
            )
        else:
            dH_out = 0.0

        # Energy balance on lumped thermal mass (dominant cp)
        dE = (Qdot_ext * dt) - dQ_des + dH_in - dH_out
        T_new = T_prev + dE / C_cp
        T_new = min(T_new, T_steam_in_K)

        if abs(T_new - T_guess) < 1e-5:
            T_guess = T_new
            break
        # Under-relaxation for stability
        T_guess = 0.5 * T_guess + 0.5 * T_new

    T_next = min(T_guess, T_steam_in_K)

    # ---- Finalize step with converged T_next
    p_sat = p_sat_water_Pa(T_next)
    n_h2o_eq = p_sat * V_ch / (R * T_next)
    n_h2o_gas = min(n_h2o_star, n_h2o_eq)

    dm_cond_step = max(0.0, (n_h2o_star - n_h2o_gas) * M_H2O)
    dH_cond_final = dm_cond_step * h_fg_water_J_per_kg(T_next)
    m_cond_liq += dm_cond_step
    mdot_cond_step = dm_cond_step
    qdot_cond[k + 1] = dH_cond_final / dt

    n_air_pre = n_air_star
    n_h2o_pre = n_h2o_gas
    n_co2_pre = n_co2_star
    n_tot_pre = n_air_pre + n_h2o_pre + n_co2_pre

    if venting_active:
        p_limit_final = max(p_set, p_sat)
        n_tot_set = p_limit_final * V_ch / (R * T_next)
        n_out = max(0.0, n_tot_pre - n_tot_set)
    else:
        n_out = 0.0

    if n_tot_pre > 0.0:
        y_air = n_air_pre / n_tot_pre
        y_h2o = n_h2o_pre / n_tot_pre
        y_co2 = n_co2_pre / n_tot_pre
    else:
        y_air = y_h2o = y_co2 = 0.0

    n_air_out = y_air * n_out
    n_h2o_out = y_h2o * n_out
    n_co2_out = y_co2 * n_out
    h_air_mol, h_h2o_mol, h_co2_mol = molar_enthalpies_J_per_mol(T_next)
    dH_out_final = (
        n_air_out * h_air_mol
        + n_h2o_out * h_h2o_mol
        + n_co2_out * h_co2_mol
    )
    qdot_out[k + 1] = dH_out_final / dt

    # Update moles after vent
    n_air = n_air_pre - n_air_out
    n_h2o = n_h2o_pre - n_h2o_out
    n_co2 = n_co2_pre - n_co2_out

    # Ensure chamber vapor stays saturated by re-evaporating condensate if needed
    n_h2o_deficit = max(0.0, n_h2o_eq - n_h2o)
    if n_h2o_deficit > 0.0:
        available_moles = m_cond_liq / M_H2O
        delta_evap = min(n_h2o_deficit, available_moles)
        if delta_evap > 0.0:
            n_h2o += delta_evap
            evap_mass = delta_evap * M_H2O
            m_cond_liq -= evap_mass
            latent_evap = evap_mass * h_fg_water_J_per_kg(T_next)
            qdot_cond[k + 1] -= latent_evap / dt
            mdot_cond[k + 1] = (mdot_cond_step - evap_mass) / dt
        else:
            mdot_cond[k + 1] = mdot_cond_step / dt
    else:
        mdot_cond[k + 1] = mdot_cond_step / dt

    # Outflow mass rates (kg/s)
    mdot_out_tot[k + 1] = (n_out * (y_air * M_air + y_h2o * M_H2O + y_co2 * M_CO2)) / dt
    mdot_out_air[k + 1] = (n_air_out * M_air) / dt
    mdot_out_h2o[k + 1] = (n_h2o_out * M_H2O) / dt
    mdot_out_steam[k + 1] = mdot_out_h2o[k + 1]
    mdot_out_co2[k + 1] = (n_co2_out * M_CO2) / dt

    # Update energy storage term (lumped thermal mass)
    qdot_heating_chamber[k + 1] = C_cp * (T_next - T_prev) / dt
    qdot_balance_res[k + 1] = (
        qdot_ext_series[k + 1]
        + qdot_steam_in[k + 1]
        - qdot_des[k + 1]
        - qdot_out[k + 1]
        - qdot_heating_chamber[k + 1]
    )

    # Update temperature
    T_prev = T_next

# -----------------------------
# 6) Plotting
# -----------------------------
# Convert for nicer axes
T_C = T - 273.15
p_mbar = p_tot / 100.0
p_air_mbar = p_air / 100.0
p_h2o_mbar = p_h2o / 100.0
p_co2_mbar = p_co2 / 100.0
p_sat_chamber_mbar = p_sat_chamber / 100.0
p_sat_in_mbar = np.full(N, p_sat_in_Pa / 100.0)
phi_plot = np.clip(phi_rel, 0.0, 1.2)

# (1) Total + partial pressures
plt.figure()
plt.plot(t, p_mbar, label="p_total [mbar]")
plt.plot(t, p_air_mbar, label="p_air [mbar]")
plt.plot(t, p_h2o_mbar, label="p_H2O [mbar]")
plt.plot(t, p_co2_mbar, label="p_CO2 [mbar]")
plt.axhline(p_set_mbar, linestyle="--", label="p_set [mbar]")
plt.xlabel("time [s]")
plt.ylabel("pressure [mbar]")
plt.legend()
plt.tight_layout()

# (2) Temperature
plt.figure()
plt.plot(t, T_C, label="T_chamber [°C]")
plt.axhline(T_steam_in_C, linestyle="--", label="T_steam_in [°C]")
plt.xlabel("time [s]")
plt.ylabel("temperature [°C]")
plt.legend()
plt.tight_layout()

# (3) Steam saturation in vs. out
plt.figure()
plt.plot(t, p_sat_chamber_mbar, label="p_sat_out (chamber) [mbar]")
plt.plot(t, p_sat_in_mbar, linestyle="--", label="p_sat_in (inlet steam) [mbar]")
plt.plot(t, p_h2o_mbar, label="p_H2O actual [mbar]")
plt.xlabel("time [s]")
plt.ylabel("pressure [mbar]")
plt.legend()
plt.tight_layout()

# (4) Steam saturation ratio φ = p_H2O / p_sat
plt.figure()
plt.plot(t, phi_plot, label="φ(t)")
plt.axhline(1.0, color="k", linestyle="--", label="φ = 1")
plt.xlabel("time [s]")
plt.ylabel("saturation ratio [-]")
plt.legend()
plt.tight_layout()

# (5) Mass flow rates in / condensed / out (total)
plt.figure()
plt.plot(t, mdot_in_steam, label="mdot_steam_in [kg/s]")
plt.plot(t, mdot_in_co2, label="mdot_CO2_des [kg/s]")
plt.plot(t, mdot_cond, label="mdot_condensed [kg/s]")
plt.plot(t, mdot_out_tot, label="mdot_out_total [kg/s]")
plt.plot(t, mdot_out_steam, label="mdot_steam_out [kg/s]")
plt.xlabel("time [s]")
plt.ylabel("mass flow [kg/s]")
plt.legend()
plt.tight_layout()

# (6) Outflow split by components
plt.figure()
plt.plot(t, mdot_out_air, label="mdot_out_air [kg/s]")
plt.plot(t, mdot_out_h2o, label="mdot_out_H2O [kg/s]")
plt.plot(t, mdot_out_co2, label="mdot_out_CO2 [kg/s]")
plt.xlabel("time [s]")
plt.ylabel("mass flow out [kg/s]")
plt.legend()
plt.tight_layout()

# (7) Energy balance rates
plt.figure()
plt.plot(t, qdot_steam_in, label="Qdot_steam_in [W]")
plt.plot(t, qdot_ext_series, label="Qdot_ext [W]")
plt.plot(t, qdot_cond, label="Qdot_cond (internal) [W]")
plt.plot(t, -qdot_des, label="Qdot_des (loss) [W]")
plt.plot(t, -qdot_out, label="Qdot_out (vent loss) [W]")
plt.plot(t, qdot_heating_chamber, label="Qdot_heating_chamber [W]")
plt.plot(t, qdot_balance_res, linestyle="--", label="Balance residual [W]")
plt.xlabel("time [s]")
plt.ylabel("energy rate [W]")
plt.legend()
plt.tight_layout()

# (8) Internal energy of chamber (relative to initial state)
plt.figure()
U_chamber = C_cp * (T - T[0])  # J
plt.plot(t, U_chamber / 1e6)
plt.xlabel("time [s]")
plt.ylabel("chamber internal energy [MJ]")
plt.title("Internal Energy Stored in Chamber")
plt.tight_layout()

plt.show()

# -----------------------------
# 9) Quick printed summary
# -----------------------------
# Time when pressure first reaches setpoint
idx_set = np.argmax(p_mbar >= p_set_mbar)
if p_mbar[idx_set] >= p_set_mbar:
    print(f"p_set reached at t = {t[idx_set]:.1f} s, T = {T_C[idx_set]:.2f} °C, p = {p_mbar[idx_set]:.2f} mbar")
else:
    print("p_set not reached within simulated time.")

# Time when temperature reaches 85°C (if it does)
idx_85 = np.argmax(T_C >= 85.0)
if T_C[idx_85] >= 85.0:
    print(f"T reaches 85°C at t = {t[idx_85]:.1f} s, p = {p_mbar[idx_85]:.2f} mbar")
else:
    print("85°C not reached within simulated time.")
