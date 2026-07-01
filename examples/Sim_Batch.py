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
# 3) Batch operating conditions
# %%
mono_0 = 8000.0          # mol/m^3
ini_0 = 50.0             # mol/m^3
CTA_0 = 50.0             # mol/m^3
T_0 = 200 + 273.15       # K

t_end = 500.0             # s
dt = 0.01                # s


# %%
# 4) Batch heat transfer / physical parameters
# %%
Tc_const_batch = 200 + 273.15  # K

P_assu_batch = 3000 * 1E5      # Pa
T_assu_batch = 150 + 273.15    # K
C_assu_batch = P_assu_batch / R_gas / T_assu_batch

rho_batch = Mw_mono * C_assu_batch  # g/m^3

Cp_C2 = 42.9                        # J/mol/K
Cp_ass_batch = Cp_C2 / Mw_mono      # J/g/K

U_heat_batch = 400                  # J/s/m^2/K
D_batch = 0.05                      # m


# %%
# 5) Define Batch model object
# %%
batch_model = PolyRxn(
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
# 6) Set Batch physical parameters
# %%
batch_model.set_batch_params(
    Tc_const=Tc_const_batch,
    rho=rho_batch,
    Cp_ass=Cp_ass_batch,
    U_heat=U_heat_batch,
    D=D_batch,
)


# %%
# 7) Run Batch simulation
# %%
batch_res = batch_model.run_batch(
    mono_0=mono_0,
    ini_0=ini_0,
    CTA_0=CTA_0,
    T_0=T_0,
    t_end=t_end,
    dt=dt,
)


# %%
# 8) Print final result
# %%
print("\n========== Batch result ==========")
print(f"Mn_final   = {batch_res['Mn_final']:.4e} g/mol")
print(f"Mw_final   = {batch_res['Mw_final']:.4e} g/mol")
print(f"PDI_final  = {batch_res['PDI_final']:.4f}")
print(f"T_final    = {batch_res['T_final']:.2f} K")
print(f"mono_final = {batch_res['mono_final']:.4e} mol/m^3")
print(f"ini_final  = {batch_res['ini_final']:.4e} mol/m^3")
print(f"CTA_final  = {batch_res['CTA_final']:.4e} mol/m^3")


# %%
# 9) Plot Batch result
# %%
plt.figure()
plt.plot(batch_res["t"], batch_res["T"], label="Batch")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(batch_res["t"], batch_res["Mn"], label="Batch")
plt.xlabel("Time [s]")
plt.ylabel("Mn [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(batch_res["t"], batch_res["Mw"], label="Batch")
plt.xlabel("Time [s]")
plt.ylabel("Mw [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(batch_res["t"], batch_res["PDI"], label="Batch")
plt.xlabel("Time [s]")
plt.ylabel("PDI [-]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(batch_res["t"], batch_res["CTA"], label="CTA")
plt.xlabel("Time [s]")
plt.ylabel("CTA concentration [mol/m³]")
plt.legend()
plt.tight_layout()
plt.show()
# %%
