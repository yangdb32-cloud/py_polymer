# %%
# Importing
# %%
from pyPoly import PolyRxn
import numpy as np
from scipy.integrate import solve_ivp
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


# %%
# 2) Common physical parameters
# %%
dH_p = -100 * 1000  # J/mol
Mw_mono = 28.05     # g/mol
R_gas = 8.3145      # J/mol/K


# %%
# 3) PFR operating conditions
# %%
mono_0 = 8000.0       # mol/m^3
ini_0 = 50.0          # mol/m^3
CTA_0 = 50.0          # mol/m^3
T_0 = 200 + 273.15    # K
Tc_in = 200 + 273.15  # K

t_end = 20.0          # s

N = 200               # number of axial grid points
h_r = 0.5             # reactor heat transfer term
h_j = 0.5             # jacket heat transfer term


# %%
# 4) PFR geometry / flow parameters
# %%
L_pfr = 10.0          # m
D_pfr = 0.05          # m
D_j_pfr = 0.08        # m
v_pfr = 0.5           # m/s
v_c_pfr = 0.5         # m/s


# %%
# 5) PFR physical parameters
# %%
P_assu_pfr = 3000 * 1E5
T_assu_pfr = 150 + 273.15
C_assu_pfr = P_assu_pfr / R_gas / T_assu_pfr

rho_pfr = Mw_mono * C_assu_pfr  # g/m^3

Cp_C2 = 42.9                    # J/mol/K
Cp_pfr = Cp_C2 / Mw_mono        # J/g/K

rho_c_pfr = 1000.0
Cp_c_pfr = 4180.0


# %%
# 6) PFR Arrhenius parameters
# 현재는 Batch/CSTR와 같은 속도상수를 쓰기 위해 Ea = 0으로 설정
# %%
A_kd = kd_test
Ea_kd = 0.0

A_kp = kp_test
Ea_kp = 0.0

A_ktc = ktc_test
Ea_ktc = 0.0

A_ktd = ktd_test
Ea_ktd = 0.0

A_ktrm = ktrm_test
Ea_ktrm = 0.0

A_ktrp = ktrp_test
Ea_ktrp = 0.0

A_kca = kca_test
Ea_kca = 0.0


# %%
# 7) PFR pressure correction parameters
# %%
P_ref_bar = 3000.0
DV_kp = 0.0


# %%
# 8) Define PFR model object
# %%
pfr_model = PolyRxn(
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
# 9) Set PFR parameters
# %%
pfr_model.set_pfr_params(
    # PFR geometry / flow parameters
    L=L_pfr,
    D=D_pfr,
    D_j=D_j_pfr,
    v=v_pfr,
    v_c=v_c_pfr,

    # PFR physical parameters
    rho_pfr=rho_pfr,
    Cp_pfr=Cp_pfr,
    rho_c=rho_c_pfr,
    Cp_c=Cp_c_pfr,

    # Arrhenius parameters
    A_kd=A_kd,
    Ea_kd=Ea_kd,

    A_kp=A_kp,
    Ea_kp=Ea_kp,

    A_ktc=A_ktc,
    Ea_ktc=Ea_ktc,

    A_ktd=A_ktd,
    Ea_ktd=Ea_ktd,

    A_ktrm=A_ktrm,
    Ea_ktrm=Ea_ktrm,

    A_ktrp=A_ktrp,
    Ea_ktrp=Ea_ktrp,

    A_kca=A_kca,
    Ea_kca=Ea_kca,

    # Pressure correction
    P_ref_bar=P_ref_bar,
    DV_kp=DV_kp,
)


# %%
# 10) Initial condition for PFR
# %%
NV = pfr_model.NV_pfr
z = np.linspace(0.0, pfr_model.L, N)

y0_pfr = np.zeros(N * NV)
s0 = y0_pfr.reshape(N, NV)

# State variables
# 0 lambda0
# 1 lambda1
# 2 lambda2
# 3 mu0
# 4 mu1
# 5 mu2
# 6 initiator
# 7 monomer
# 8 CTA
# 9 reactor temperature
# 10 coolant temperature

s0[:, 6] = ini_0
s0[:, 7] = mono_0
s0[:, 8] = CTA_0
s0[:, 9] = T_0
s0[:, 10] = Tc_in


# %%
# 11) Build PFR ODE function
# %%
pfr_rhs = pfr_model.pfr_odes(
    ini_0=ini_0,
    mono_0=mono_0,
    CTA_0=CTA_0,
    T_0=T_0,
    Tc_in=Tc_in,
    h_r=h_r,
    h_j=h_j,
    kp_factor=1.0,
    N=N,
)

jac_sp = pfr_model.build_jac_sparsity(N, NV)
atol = np.tile(pfr_model.atol_per_var_pfr, N)


# %%
# 12) Run PFR simulation
# %%
pfr_sol = solve_ivp(
    pfr_rhs,
    t_span=(0.0, t_end),
    y0=y0_pfr,
    method="BDF",
    atol=atol,
    rtol=1e-6,
    jac_sparsity=jac_sp,
)


# %%
# 13) Extract final axial profile
# %%
pfr_final = pfr_sol.y[:, -1].reshape(N, NV)

lamb0_pfr = pfr_final[:, 0]
lamb1_pfr = pfr_final[:, 1]
lamb2_pfr = pfr_final[:, 2]

mu0_pfr = pfr_final[:, 3]
mu1_pfr = pfr_final[:, 4]
mu2_pfr = pfr_final[:, 5]

ini_pfr = pfr_final[:, 6]
mono_pfr = pfr_final[:, 7]
CTA_pfr = pfr_final[:, 8]
T_pfr = pfr_final[:, 9]
Tc_pfr = pfr_final[:, 10]


# %%
# 14) Calculate Mn, Mw, PDI
# %%
eps = 1E-30

Mn_pfr = (
    pfr_model.Mw_mono
    * (lamb1_pfr + mu1_pfr + eps)
    / (lamb0_pfr + mu0_pfr + eps)
)

Mw_pfr = (
    pfr_model.Mw_mono
    * (lamb2_pfr + mu2_pfr + eps)
    / (lamb1_pfr + mu1_pfr + eps)
)

PDI_pfr = Mw_pfr / Mn_pfr


# %%
# 15) Save result dictionary
# %%
pfr_res = {
    "mode": "PFR",
    "z": z,
    "Y_final": pfr_final,

    "lambda0": lamb0_pfr,
    "lambda1": lamb1_pfr,
    "lambda2": lamb2_pfr,

    "mu0": mu0_pfr,
    "mu1": mu1_pfr,
    "mu2": mu2_pfr,

    "ini": ini_pfr,
    "mono": mono_pfr,
    "CTA": CTA_pfr,
    "T": T_pfr,
    "Tc": Tc_pfr,

    "Mn": Mn_pfr,
    "Mw": Mw_pfr,
    "PDI": PDI_pfr,

    "Mn_final": Mn_pfr[-1],
    "Mw_final": Mw_pfr[-1],
    "PDI_final": PDI_pfr[-1],
    "T_final": T_pfr[-1],
    "mono_final": mono_pfr[-1],
    "ini_final": ini_pfr[-1],
    "CTA_final": CTA_pfr[-1],
}


# %%
# 16) Print final result
# %%
print("\n========== PFR result ==========")
print(f"success     = {pfr_sol.success}")
print(f"message     = {pfr_sol.message}")
print(f"Mn_outlet   = {pfr_res['Mn_final']:.4e} g/mol")
print(f"Mw_outlet   = {pfr_res['Mw_final']:.4e} g/mol")
print(f"PDI_outlet  = {pfr_res['PDI_final']:.4f}")
print(f"T_outlet    = {pfr_res['T_final']:.2f} K")
print(f"mono_outlet = {pfr_res['mono_final']:.4e} mol/m^3")
print(f"ini_outlet  = {pfr_res['ini_final']:.4e} mol/m^3")
print(f"CTA_outlet  = {pfr_res['CTA_final']:.4e} mol/m^3")


# %%
# 17) Plot PFR axial profiles
# %%
plt.figure()
plt.plot(pfr_res["z"], pfr_res["T"], label="Reactor")
plt.plot(pfr_res["z"], pfr_res["Tc"], label="Coolant")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pfr_res["z"], pfr_res["Mn"], label="PFR")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Mn [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pfr_res["z"], pfr_res["Mw"], label="PFR")
plt.xlabel("Reactor length z [m]")
plt.ylabel("Mw [g/mol]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pfr_res["z"], pfr_res["PDI"], label="PFR")
plt.xlabel("Reactor length z [m]")
plt.ylabel("PDI [-]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pfr_res["z"], pfr_res["CTA"], label="CTA")
plt.xlabel("Reactor length z [m]")
plt.ylabel("CTA concentration [mol/m³]")
plt.legend()
plt.tight_layout()
plt.show()