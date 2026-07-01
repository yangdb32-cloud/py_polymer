# %%
# Importing
# %%
from pyPoly import PolyRxn
import numpy as np
import matplotlib.pyplot as plt


# %%
# 1) Common kinetic parameters
# %%
kd_test = 0.8
kp_test = 15
ktc_test = 0.02
ktd_test = 0.02
ktrm_test = 15E-3
ktrp_test = 10E-3
kca_test = 5E-3       # CTA chain-transfer coefficient
f_test = 0.8          # Initiator efficiency

par_list = [
    kd_test,
    kp_test,
    ktc_test,
    ktd_test,
    ktrm_test,
    ktrp_test,
    kca_test,
    f_test,
]


# %%
# 2) Common physical parameters
# %%
dH_p = -100 * 1000  # J/mol
Mw_mono = 28.05     # g/mol
R_gas = 8.3145      # J/mol/K


# %%
# 3) CSTR operating conditions
# %%
mono_0 = 8000.0          # mol/m^3
ini_0 = 50.0             # mol/m^3
CTA_0 = 50.0             # mol/m^3
T_0 = 200 + 273.15       # K

tau_cstr = 20.0           # s
t_end = 500.0             # s
dt = 0.01                # s


# %%
# 4) CSTR heat transfer / physical parameters
# %%
Tc_const_cstr = 200 + 273.15  # K

P_assu_cstr = 3000 * 1E5      # Pa
T_assu_cstr = 150 + 273.15    # K
C_assu_cstr = P_assu_cstr / R_gas / T_assu_cstr

rho_cstr = Mw_mono * C_assu_cstr  # g/m^3

Cp_C2 = 42.9                      # J/mol/K
Cp_ass_cstr = Cp_C2 / Mw_mono     # J/g/K

U_heat_cstr = 400                 # J/s/m^2/K
D_cstr = 0.05                     # m


# %%
# 5) Define CSTR model object
# %%
cstr_model = PolyRxn(
    kd=kd_test,
    kp=kp_test,
    ktc=ktc_test,
    ktd=ktd_test,
    ktrm=ktrm_test,
    ktrp=ktrp_test,
    kca=kca_test,
    f=f_test,
    dH_p=dH_p,
    Mw_mono=Mw_mono,
    R_gas=R_gas,
)


# %%
# 6) Set CSTR physical parameters
# %%
cstr_model.set_cstr_params(
    Tc_const=Tc_const_cstr,
    rho=rho_cstr,
    Cp_ass=Cp_ass_cstr,
    U_heat=U_heat_cstr,
    D=D_cstr,
)


# %%
# 7) Run CSTR simulation
# %%
cstr_res = cstr_model.run_cstr(
    mono_0=mono_0,
    ini_0=ini_0,
    CTA_0=CTA_0,
    T_0=T_0,
    tau=tau_cstr,
    t_end=t_end,
    dt=dt,
)


# %%
# 8) Print final result
# %%
print("\n========== CSTR result ==========")
print(f"tau        = {cstr_res['tau']:.4f} s")
print(f"Mn_final   = {cstr_res['Mn_final']:.4e} g/mol")
print(f"Mw_final   = {cstr_res['Mw_final']:.4e} g/mol")
print(f"PDI_final  = {cstr_res['PDI_final']:.4f}")
print(f"T_final    = {cstr_res['T_final']:.2f} K")
print(f"mono_final = {cstr_res['mono_final']:.4e} mol/m^3")
print(f"ini_final  = {cstr_res['ini_final']:.4e} mol/m^3")
print(f"CTA_final  = {cstr_res['CTA_final']:.4e} mol/m^3")


# %%
# 9) Plot CSTR result
# %%
plt.figure()
plt.plot(cstr_res["t"], cstr_res["T"], label="CSTR")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cstr_res["t"], cstr_res["Mn"], label="CSTR")
plt.xlabel("Time [s]")
plt.ylabel("Mn [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cstr_res["t"], cstr_res["Mw"], label="CSTR")
plt.xlabel("Time [s]")
plt.ylabel("Mw [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cstr_res["t"], cstr_res["PDI"], label="CSTR")
plt.xlabel("Time [s]")
plt.ylabel("PDI [-]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cstr_res["t"], cstr_res["CTA"], label="CTA")
plt.xlabel("Time [s]")
plt.ylabel("CTA concentration [mol/m^3]")
plt.legend()
plt.tight_layout()
plt.show()