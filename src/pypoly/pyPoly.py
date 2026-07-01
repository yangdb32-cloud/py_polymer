# %%
# Polymerization package tset
# BATHC, CSTR, PFR simualtions
# %%
# Importing Packages
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
import time

# %%
# Initial conditions
kd_test = 0.8
kp_test = 15
ktc_test = 0.02
ktd_test = 0.02
ktrm_test = 15E-3
ktrp_test = 10E-3
kca_test = 5E-3   # CTA chain transfer coefficient
# kp >> ktrm > kd >> ktc, ktd, ktrp 순서여야함
# 만약 kp = 5이고 ktrm = ktrp = 1 m³/(mol·s) = 1000 L/(mol·s) 이라면 비현실적...
# → kp와 같은 수준으로 너무 높음 (실제로는 kp의 1/1000 ~ 1/100 수준)
f_test = 0.8 # Initiator efficiency (80% assumed)

par_list = [kd_test, kp_test, ktc_test, ktd_test, 
            ktrm_test, ktrp_test, kca_test, f_test,]


Tc_const = 200+273 # K
dH_p = -100*1000 # 100 kJ/mol = 100,000 J/mol
Mw_mono = 28.05# g/mol

# Dnesity (rho)
P_assu = 3000*1E5 # 3000 bar (1E8 Pa) & 150+273 K
T_assu = 150+273  # K
R_gas = 8.3145    # J/mol/K
C_assu = P_assu/R_gas/T_assu # mol/m^3
rho = Mw_mono*C_assu # g/m^3
#print('rho =')
#print(rho)

# Heat capacity Cp
Cp_C2 = 42.9  # J/mol/K
Cp_ass = Cp_C2/Mw_mono  # J/g/K

# Overall heat transfer coefficients (U)
U_heat = 400    # J/s/m^2/K

# Diameter
D = 0.05       # 10 cm = 0.10 m 


# %%
class PolyRxn:
    def __init__(
        self,
        # Common kinetic parameters
        kd, kp, ktc, ktd, ktrm, ktrp, kca, f,

        # Common physical parameters
        dH_p,
        Mw_mono,

        # Common parameters
        R_gas=8.3145,
    ):
        # ============================================================
        # Common kinetic parameters
        # ============================================================
        self.kd = kd
        self.kp = kp
        self.ktc = ktc
        self.ktd = ktd
        self.ktrm = ktrm
        self.ktrp = ktrp
        self.kca = kca
        self.f = f

        # ============================================================
        # Common physical parameters
        # ============================================================
        self.dH_p = dH_p
        self.Mw_mono = Mw_mono

        # ============================================================
        # Common
        # ============================================================
        self.R_gas = R_gas

        # Flags for model-specific parameter settings
        self.is_batch_params_set = False
        self.is_cstr_params_set = False
        self.is_pfr_params_set = False

    def set_batch_params(
        self,
        Tc_const,
        rho,
        Cp_ass,
        U_heat,
        D,
    ):
        """Set physical parameters required only for Batch simulations."""
        self.Tc_const_batch = Tc_const
        self.rho_batch = rho
        self.Cp_ass_batch = Cp_ass
        self.U_heat_batch = U_heat
        self.D_batch = D
        self.is_batch_params_set = True

    def set_cstr_params(
        self,
        Tc_const,
        rho,
        Cp_ass,
        U_heat,
        D,
    ):
        """Set physical parameters required only for CSTR simulations."""
        self.Tc_const_cstr = Tc_const
        self.rho_cstr = rho
        self.Cp_ass_cstr = Cp_ass
        self.U_heat_cstr = U_heat
        self.D_cstr = D
        self.is_cstr_params_set = True

    def set_pfr_params(
        self,
        # PFR geometry / flow parameters
        L,
        D,
        D_j,
        v,
        v_c,

        # PFR physical parameters
        rho_pfr,
        Cp_pfr,
        rho_c,
        Cp_c,

        # PFR Arrhenius parameters
        A_kd,
        Ea_kd,
        A_kp,
        Ea_kp,
        A_ktc,
        Ea_ktc,
        A_ktd,
        Ea_ktd,
        A_ktrm,
        Ea_ktrm,
        A_ktrp,
        Ea_ktrp,
        A_kca,
        Ea_kca,

        # PFR pressure correction
        P_ref_bar,
        DV_kp,
    ):
        """Set parameters required only for PFR simulations."""
        # ============================================================
        # PFR geometry / flow parameters
        # ============================================================
        self.L = L
        self.D = D
        self.D_j = D_j
        self.v = v
        self.v_c = v_c

        self.A_r = np.pi / 4 * self.D**2
        self.A_c = np.pi / 4 * (self.D_j**2 - self.D**2)

        # ============================================================
        # PFR physical parameters
        # ============================================================
        self.rho_pfr = rho_pfr
        self.Cp_pfr = Cp_pfr
        self.rho_c = rho_c
        self.Cp_c = Cp_c

        # ============================================================
        # PFR Arrhenius parameters
        # ============================================================
        self.A_kd = A_kd
        self.Ea_kd = Ea_kd

        self.A_kp = A_kp
        self.Ea_kp = Ea_kp

        self.A_ktc = A_ktc
        self.Ea_ktc = Ea_ktc

        self.A_ktd = A_ktd
        self.Ea_ktd = Ea_ktd

        self.A_ktrm = A_ktrm
        self.Ea_ktrm = Ea_ktrm

        self.A_ktrp = A_ktrp
        self.Ea_ktrp = Ea_ktrp

        self.A_kca = A_kca
        self.Ea_kca = Ea_kca

        # ============================================================
        # PFR pressure correction
        # ============================================================
        self.P_ref_bar = P_ref_bar
        self.DV_kp = DV_kp

        # ============================================================
        # PFR solver settings
        # ============================================================
        self.NV_pfr = 11

        self.atol_per_var_pfr = np.array([
            1e-14, 1e-11, 1e-7,
            1e-11, 1e-8, 1e-3,
            1e-8, 1e-1, 1e-4, 1e-4, 1e-4
        ])

        # PFR에서는 f_eff라는 이름 사용
        self.f_eff = self.f
        self.is_pfr_params_set = True

    def check_batch_ready(self):
        """Check whether Batch-specific parameters are set."""
        if not self.is_batch_params_set:
            raise ValueError(
                "Batch simulation requires physical parameters. "
                "Please call set_batch_params() first."
            )

    def check_cstr_ready(self):
        """Check whether CSTR-specific parameters are set."""
        if not self.is_cstr_params_set:
            raise ValueError(
                "CSTR simulation requires physical parameters. "
                "Please call set_cstr_params() first."
            )

    def check_pfr_ready(self):
        """Check whether PFR-specific parameters are set."""
        if not self.is_pfr_params_set:
            raise ValueError(
                "PFR simulation requires PFR-specific parameters. "
                "Please call set_pfr_params() first."
            )

    # Batch
    def batch_odes(self, y, t):
        lamb0 = y[0]
        lamb1 = y[1]
        lamb2 = y[2]
        mu0 = y[3]
        mu1 = y[4]
        mu2 = y[5]
        ini = y[6]
        mono = y[7]
        CTA = y[8]
        T = y[9]

        # kinetic parameters from self
        kd = self.kd
        kp = self.kp
        ktc = self.ktc
        ktd = self.ktd
        ktrm = self.ktrm
        ktrp = self.ktrp
        kca = self.kca
        f = self.f

        # mu3 from Schulz-Zimm distribution
        # Hulburt & Katz approximation
        mu3_guess = mu2 * (2 * mu0 * mu2 - mu1**2) / (mu0 + 1E-6) / (mu1 + 1E-6)
        mu3 = max(mu3_guess, 0)

        # Initiator and monomer consumption
        dini_dt = -kd * ini
        dmono_dt = -kp * mono * lamb0
        dCTA_dt = -kca * CTA * lamb0

        # Growth radical moment
        dlamb0_dt = 2 * f * kd * ini - (ktc + ktd) * lamb0**2

        term1_1 = kp * mono * lamb0
        term1_2 = ktrm * mono * (lamb0 - lamb1)
        term1_3 = ktrp * (lamb0 * mu2 - lamb1 * mu1)
        term1_4 = -(ktc + ktd) * lamb0 * lamb1
        term1_5 = kca * CTA * (lamb0 - lamb1)
        dlamb1_dt = term1_1 + term1_2 + term1_3 + term1_4 + term1_5

        term2_1 = kp * mono * (2 * lamb1 + lamb0)
        term2_2 = ktrm * mono * (lamb0 - lamb2)
        term2_3 = ktrp * (lamb0 * mu3 - lamb2 * mu1)
        term2_4 = -(ktc + ktd) * lamb0 * lamb2
        term2_5 = kca * CTA * (lamb0 - lamb2)
        dlamb2_dt = term2_1 + term2_2 + term2_3 + term2_4 + term2_5

        # Growth of dead polymer moment
        term3_1 = ktrm * mono * lamb0
        term3_2 = (0.5 * ktc + ktd) * lamb0**2
        term3_3 = kca * CTA * lamb0
        dmu0_dt = term3_1 + term3_2 + term3_3

        dmu1_dt = ktrm * mono * lamb1 + (ktc + ktd) * lamb0 * lamb1
        dmu1_dt += ktrp * (lamb1 * mu1 - lamb0 * mu2)
        dmu1_dt += kca * CTA * lamb1

        term4_1 = ktrm * mono * lamb2
        term4_2 = ktd * lamb0 * lamb2
        term4_3 = ktc * (lamb0 * lamb2 + lamb1**2)
        term4_4 = kca * CTA * lamb2
        dmu2_dt = term4_1 + term4_2 + term4_3 + term4_4
        dmu2_dt += ktrp * (lamb2 * mu1 - lamb0 * mu3)

        # Energy balance
        dT_dt = (
            (-self.dH_p) / self.rho_batch / self.Cp_ass_batch * kp * mono * lamb0
            - 4 * self.U_heat_batch / self.rho_batch / self.Cp_ass_batch / self.D_batch
            * (T - self.Tc_const_batch)
        )

        dydt_list = [
            dlamb0_dt,
            dlamb1_dt,
            dlamb2_dt,
            dmu0_dt,
            dmu1_dt,
            dmu2_dt,
            dini_dt,
            dmono_dt,
            dCTA_dt,
            dT_dt,
        ]

        dy_dt = np.array(dydt_list)

        return dy_dt


    def run_batch(
        self,
        mono_0,
        ini_0,
        CTA_0,
        T_0,
        t_end,
        dt,
    ):
        self.check_batch_ready()

        y0 = np.zeros(10)

        # state variables
        # y[6] = initiator
        # y[7] = monomer
        # y[8] = CTA
        # y[9] = temperature
        y0[6] = ini_0
        y0[7] = mono_0
        y0[8] = CTA_0
        y0[9] = T_0

        t_ran = np.arange(0, t_end + dt, dt)

        y_res = odeint(
            self.batch_odes,
            y0,
            t_ran,
        )

        lamb0_res = y_res[:, 0]
        lamb1_res = y_res[:, 1]
        lamb2_res = y_res[:, 2]
        mu0_res = y_res[:, 3]
        mu1_res = y_res[:, 4]
        mu2_res = y_res[:, 5]
        ini_res = y_res[:, 6]
        mono_res = y_res[:, 7]
        CTA_res = y_res[:, 8]
        T_res = y_res[:, 9]

        eps = 1E-30

        Mn_res = self.Mw_mono * (lamb1_res + mu1_res + eps) / (lamb0_res + mu0_res + eps)
        Mw_res = self.Mw_mono * (lamb2_res + mu2_res + eps) / (lamb1_res + mu1_res + eps)
        PDI_res = Mw_res / Mn_res

        result = {
            "mode": "BATCH",
            "t": t_ran,
            "Y": y_res,

            "lambda0": lamb0_res,
            "lambda1": lamb1_res,
            "lambda2": lamb2_res,

            "mu0": mu0_res,
            "mu1": mu1_res,
            "mu2": mu2_res,

            "ini": ini_res,
            "mono": mono_res,
            "CTA": CTA_res,
            "T": T_res,

            "Mn": Mn_res,
            "Mw": Mw_res,
            "PDI": PDI_res,

            "Mn_final": Mn_res[-1],
            "Mw_final": Mw_res[-1],
            "PDI_final": PDI_res[-1],
            "T_final": T_res[-1],
            "mono_final": mono_res[-1],
            "CTA_final": CTA_res[-1],
            "ini_final": ini_res[-1],
        }

        return result

    # CSTR
    def cstr_odes(self, y, t, y_in, tau):
        lamb0 = y[0]
        lamb1 = y[1]
        lamb2 = y[2]
        mu0 = y[3]
        mu1 = y[4]
        mu2 = y[5]
        ini = y[6]
        mono = y[7]
        CTA = y[8]
        T = y[9]

        # kinetic parameters from self
        kd = self.kd
        kp = self.kp
        ktc = self.ktc
        ktd = self.ktd
        ktrm = self.ktrm
        ktrp = self.ktrp
        kca = self.kca
        f = self.f

        # mu3 approximation
        mu3_guess = mu2 * (2 * mu0 * mu2 - mu1**2) / (mu0 + 1E-6) / (mu1 + 1E-6)
        mu3 = max(mu3_guess, 0)

        # Reaction terms
        dini_dt_rxn = -kd * ini
        dmono_dt_rxn = -kp * mono * lamb0
        dCTA_dt_rxn = -kca * CTA * lamb0

        dlamb0_dt_rxn = 2 * f * kd * ini - (ktc + ktd) * lamb0**2

        term1_1 = kp * mono * lamb0
        term1_2 = ktrm * mono * (lamb0 - lamb1)
        term1_3 = ktrp * (lamb0 * mu2 - lamb1 * mu1)
        term1_4 = -(ktc + ktd) * lamb0 * lamb1
        term1_5 = kca * CTA * (lamb0 - lamb1)
        dlamb1_dt_rxn = term1_1 + term1_2 + term1_3 + term1_4 + term1_5

        term2_1 = kp * mono * (2 * lamb1 + lamb0)
        term2_2 = ktrm * mono * (lamb0 - lamb2)
        term2_3 = ktrp * (lamb0 * mu3 - lamb2 * mu1)
        term2_4 = -(ktc + ktd) * lamb0 * lamb2
        term2_5 = kca * CTA * (lamb0 - lamb2)
        dlamb2_dt_rxn = term2_1 + term2_2 + term2_3 + term2_4 + term2_5

        term3_1 = ktrm * mono * lamb0
        term3_2 = (0.5 * ktc + ktd) * lamb0**2
        term3_3 = kca * CTA * lamb0
        dmu0_dt_rxn = term3_1 + term3_2 + term3_3

        dmu1_dt_rxn = ktrm * mono * lamb1 + (ktc + ktd) * lamb0 * lamb1
        dmu1_dt_rxn += ktrp * (lamb1 * mu1 - lamb0 * mu2)
        dmu1_dt_rxn += kca * CTA * lamb1

        term4_1 = ktrm * mono * lamb2
        term4_2 = ktd * lamb0 * lamb2
        term4_3 = ktc * (lamb0 * lamb2 + lamb1**2)
        term4_4 = kca * CTA * lamb2
        dmu2_dt_rxn = term4_1 + term4_2 + term4_3 + term4_4
        dmu2_dt_rxn += ktrp * (lamb2 * mu1 - lamb0 * mu3)

        dT_dt_rxn = (
            (-self.dH_p) / self.rho_cstr / self.Cp_ass_cstr
            * kp * mono * lamb0
        )

        dT_dt_cooling = (
            -4 * self.U_heat_cstr / self.rho_cstr / self.Cp_ass_cstr / self.D_cstr
            * (T - self.Tc_const_cstr)
        )

        dydt_rxn = np.array([
            dlamb0_dt_rxn,
            dlamb1_dt_rxn,
            dlamb2_dt_rxn,
            dmu0_dt_rxn,
            dmu1_dt_rxn,
            dmu2_dt_rxn,
            dini_dt_rxn,
            dmono_dt_rxn,
            dCTA_dt_rxn,
            dT_dt_rxn + dT_dt_cooling,
        ])

        # CSTR inlet-outlet term
        dydt_flow = (y_in - y) / tau

        dydt = dydt_flow + dydt_rxn

        return dydt


    def run_cstr(
        self,
        mono_0,
        ini_0,
        CTA_0,
        T_0,
        tau,
        t_end,
        dt,
    ):
        self.check_cstr_ready()

        y0 = np.zeros(10)

        # initial condition
        y0[6] = ini_0
        y0[7] = mono_0
        y0[8] = CTA_0
        y0[9] = T_0

        y_in = np.zeros(10)

        # inlet condition
        y_in[6] = ini_0
        y_in[7] = mono_0
        y_in[8] = CTA_0
        y_in[9] = T_0

        t_ran = np.arange(0, t_end + dt, dt)

        y_res = odeint(
            self.cstr_odes,
            y0,
            t_ran,
            args=(y_in, tau),
        )

        lamb0_res = y_res[:, 0]
        lamb1_res = y_res[:, 1]
        lamb2_res = y_res[:, 2]
        mu0_res = y_res[:, 3]
        mu1_res = y_res[:, 4]
        mu2_res = y_res[:, 5]
        ini_res = y_res[:, 6]
        mono_res = y_res[:, 7]
        CTA_res = y_res[:, 8]
        T_res = y_res[:, 9]

        eps = 1E-30

        Mn_res = self.Mw_mono * (lamb1_res + mu1_res + eps) / (lamb0_res + mu0_res + eps)
        Mw_res = self.Mw_mono * (lamb2_res + mu2_res + eps) / (lamb1_res + mu1_res + eps)
        PDI_res = Mw_res / Mn_res

        result = {
            "mode": "CSTR",
            "t": t_ran,
            "Y": y_res,

            "lambda0": lamb0_res,
            "lambda1": lamb1_res,
            "lambda2": lamb2_res,

            "mu0": mu0_res,
            "mu1": mu1_res,
            "mu2": mu2_res,

            "ini": ini_res,
            "mono": mono_res,
            "CTA": CTA_res,
            "T": T_res,

            "Mn": Mn_res,
            "Mw": Mw_res,
            "PDI": PDI_res,

            "Mn_final": Mn_res[-1],
            "Mw_final": Mw_res[-1],
            "PDI_final": PDI_res[-1],
            "T_final": T_res[-1],
            "mono_final": mono_res[-1],
            "CTA_final": CTA_res[-1],
            "ini_final": ini_res[-1],

            "tau": tau,
        }

        return result


    # PFR
    """
    # ---- Spatial Grid ----------------------------------------------------------
    N  = 200
    NV = 10
    z  = np.linspace(0.0, L, N)
    dz = z[1] - z[0]
    print(f'Grid: N={N}, dz={dz:.2f} m, τ_res={L/v:.1f} s')


    # ---- Jacobian Sparsity -----------------------------------------------------
    def build_jac_sparsity(N, NV=10):
        rows, cols = [], []
        for i in range(N):
            for j in range(NV):
                row = i * NV + j
                for k in range(NV):
                    rows.append(row); cols.append(i * NV + k)
                if j < 10 and i > 0:
                    rows.append(row); cols.append((i - 1) * NV + j)
                if j == 10 and i < N - 1:
                    rows.append(row); cols.append((i + 1) * NV + 10)
        n = N * NV
        return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))

    jac_sp = build_jac_sparsity(N, NV)
    print(f'Jacobian: {N*NV}×{N*NV}, nnz={jac_sp.nnz}')
    """

    # PFR
    def arrhenius(self, A, Ea, T):
        return A * np.exp(-Ea / (self.R_gas * T))


    def build_jac_sparsity(self, N, NV=None):
        self.check_pfr_ready()

        if NV is None:
            NV = self.NV_pfr

        rows, cols = [], []

        for i in range(N):
            for j in range(NV):
                row = i * NV + j

                for k in range(NV):
                    rows.append(row)
                    cols.append(i * NV + k)

                if j < 10 and i > 0:
                    rows.append(row)
                    cols.append((i - 1) * NV + j)

                if j == 10 and i < N - 1:
                    rows.append(row)
                    cols.append((i + 1) * NV + 10)

        n = N * NV

        return csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n, n)
        )


    def pfr_odes(
        self,
        ini_0,
        mono_0,
        CTA_0,
        T_0,
        Tc_in,
        h_r,
        h_j,
        kp_factor=1.0,
        N=None,
    ):
        """Return PFR ODE function for given operating conditions."""
        self.check_pfr_ready()

        if N is None:
            N = 200

        NV = self.NV_pfr
        dz = self.L / (N - 1)
        T_safe = 300.0 + 273.15

        def pfr_odes(t, y):
            s = y.reshape(N, NV)

            lamb0 = s[:, 0]
            lamb1 = s[:, 1]
            lamb2 = s[:, 2]

            mu0 = s[:, 3]
            mu1 = s[:, 4]
            mu2 = s[:, 5]

            ini = s[:, 6]
            mono = s[:, 7]
            CTA = s[:, 8]
            T = s[:, 9]
            Tc = s[:, 10]

            kd = self.arrhenius(self.A_kd, self.Ea_kd, T)
            kp = self.arrhenius(self.A_kp, self.Ea_kp, T) * kp_factor
            ktc = self.arrhenius(self.A_ktc, self.Ea_ktc, T)
            ktd = self.arrhenius(self.A_ktd, self.Ea_ktd, T)
            ktrm = self.arrhenius(self.A_ktrm, self.Ea_ktrm, T)
            ktrp = self.arrhenius(self.A_ktrp, self.Ea_ktrp, T)
            kca = self.arrhenius(self.A_kca, self.Ea_kca, T)

            eps3 = 1e-12

            valid = (mu0 > eps3) & (mu1 > eps3)

            mu3_hk = np.where(
                valid,
                mu2 * (2.0 * mu0 * mu2 - mu1**2) / (mu0 * mu1 + eps3),
                0.0
            )

            mu3_cs = np.where(
                mu1 > eps3,
                mu2**2 / (mu1 + eps3),
                0.0
            )

            mu3 = np.maximum(mu3_hk, mu3_cs)

            R_l0 = (
                2.0 * self.f_eff * kd * ini
                - (ktc + ktd) * lamb0**2
            )

            R_l1 = (
                kp * mono * lamb0
                + ktrm * mono * (lamb0 - lamb1)
                + ktrp * (lamb0 * mu2 - lamb1 * mu1)
                + kca * CTA * (lamb0 - lamb1)
                - (ktc + ktd) * lamb0 * lamb1
            )

            R_l2 = (
                kp * mono * (2.0 * lamb1 + lamb0)
                + ktrm * mono * (lamb0 - lamb2)
                + ktrp * (lamb0 * mu3 - lamb2 * mu1)
                + kca * CTA * (lamb0 - lamb2)
                - (ktc + ktd) * lamb0 * lamb2
            )

            R_m0 = (
                ktrm * mono * lamb0
                + kca * CTA * lamb0
                + (0.5 * ktc + ktd) * lamb0**2
            )

            R_m1 = (
                ktrm * mono * lamb1
                + ktrp * (lamb1 * mu1 - lamb0 * mu2)
                + kca * CTA * lamb1
                + (ktc + ktd) * lamb0 * lamb1
            )

            R_m2 = (
                ktrm * mono * lamb2
                + ktd * lamb0 * lamb2
                + ktc * (lamb0 * lamb2 + lamb1**2)
                + ktrp * (lamb2 * mu1 - lamb0 * mu3)
                + kca * CTA * lamb2
            )

            R_ini = -kd * ini
            R_mono = -kp * mono * lamb0
            R_CTA = -kca * CTA * lamb0

            RT = (
                (-self.dH_p) / (self.rho_pfr * self.Cp_pfr)
                * kp * mono * lamb0
                - h_r * (T - Tc)
            )

            RT -= np.maximum(0.0, T - T_safe) * 10.0

            R_Tc = h_j * (T - Tc)

            dydt = np.zeros_like(s)

            for j, (C, R, BC) in enumerate(zip(
                [lamb0, lamb1, lamb2, mu0, mu1, mu2, ini, mono, CTA, T],
                [R_l0, R_l1, R_l2, R_m0, R_m1, R_m2, R_ini, R_mono, R_CTA, RT],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ini_0, mono_0, CTA_0, T_0]
            )):
                C_up = np.empty(N)
                C_up[0] = BC
                C_up[1:] = C[:-1]

                dCdz = (C - C_up) / dz

                dydt[0, j] = 0.0
                dydt[1:, j] = R[1:] - self.v * dCdz[1:]

            Tc_dn = np.empty(N)
            Tc_dn[:-1] = Tc[1:]
            Tc_dn[-1] = Tc_in

            dTcdz = (Tc_dn - Tc) / dz

            dydt[:N - 1, 10] = R_Tc[:N - 1] + self.v_c * dTcdz[:N - 1]
            dydt[N - 1, 10] = 0.0

            return dydt.ravel()

        return pfr_odes



# %%
# BATCH simulation
if __name__ == "__main__":
    # ============================================================
    # Common kinetic parameters
    # ============================================================
    kd_test = 0.8
    kp_test = 15
    ktc_test = 0.02
    ktd_test = 0.02
    ktrm_test = 15E-3
    ktrp_test = 10E-3
    kca_test = 5E-3
    f_test = 0.8  # Initiator efficiency

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

    # ============================================================
    # Common physical parameters
    # ============================================================
    dH_p = -100 * 1000  # J/mol
    Mw_mono = 28.05     # g/mol
    R_gas = 8.3145      # J/mol/K

    # ============================================================
    # Batch parameters
    # ============================================================
    mono_0 = 8000.0          # mol/m^3
    ini_0 = 50.0             # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0 = 200 + 273.15       # K

    t_end = 20.0             # s
    dt = 0.01                # s

    # Batch heat transfer / physical parameters
    Tc_const_batch = 200 + 273.15  # K

    P_assu_batch = 3000 * 1E5      # Pa
    T_assu_batch = 150 + 273.15    # K
    C_assu_batch = P_assu_batch / R_gas / T_assu_batch
    rho_batch = Mw_mono * C_assu_batch  # g/m^3

    Cp_C2 = 42.9                   # J/mol/K
    Cp_ass_batch = Cp_C2 / Mw_mono # J/g/K

    U_heat_batch = 400             # J/s/m^2/K
    D_batch = 0.05                 # m




    # ---- Batch operating conditions ----------------------------
    mono_0 = 8000.0          # mol/m^3
    ini_0  = 50.0            # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0    = 200 + 273.15    # K

    t_end = 20.0             # s
    dt = 0.01                # s

    # ---- Model object for Batch --------------------------------
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
    )

    batch_model.set_batch_params(
        Tc_const=Tc_const_batch,
        rho=rho_batch,
        Cp_ass=Cp_ass_batch,
        U_heat=U_heat_batch,
        D=D_batch,
    )

    # ---- Run Batch simulation ----------------------------------
    batch_res = batch_model.run_batch(
        mono_0=mono_0,
        ini_0=ini_0,
        CTA_0=CTA_0,
        T_0=T_0,
        t_end=t_end,
        dt=dt,
    )

    # ---- Print final result ------------------------------------
    print("\n========== Batch result ==========")
    print(f"Mn_final   = {batch_res['Mn_final']:.4e} g/mol")
    print(f"Mw_final   = {batch_res['Mw_final']:.4e} g/mol")
    print(f"PDI_final  = {batch_res['PDI_final']:.4f}")
    print(f"T_final    = {batch_res['T_final']:.2f} K")
    print(f"mono_final = {batch_res['mono_final']:.4e} mol/m^3")
    print(f"ini_final  = {batch_res['ini_final']:.4e} mol/m^3")

    # ---- Plot Batch result -------------------------------------
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
# %%
# CSTR simulation
if __name__ == "__main__":
    # ============================================================
    # Common kinetic parameters
    # ============================================================
    kd_test = 0.8
    kp_test = 15
    ktc_test = 0.02
    ktd_test = 0.02
    ktrm_test = 15E-3
    ktrp_test = 10E-3
    kca_test = 5E-3
    f_test = 0.8  # Initiator efficiency

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

    # ============================================================
    # Common physical parameters
    # ============================================================
    dH_p = -100 * 1000  # J/mol
    Mw_mono = 28.05     # g/mol
    R_gas = 8.3145      # J/mol/K

    # ============================================================
    # CSTR parameters
    # ============================================================
    mono_0 = 8000.0          # mol/m^3
    ini_0 = 50.0             # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0 = 200 + 273.15       # K

    tau_cstr = 5.0           # s
    t_end = 20.0             # s
    dt = 0.01                # s

    # CSTR heat transfer / physical parameters
    Tc_const_cstr = 200 + 273.15  # K

    P_assu_cstr = 3000 * 1E5      # Pa
    T_assu_cstr = 150 + 273.15    # K
    C_assu_cstr = P_assu_cstr / R_gas / T_assu_cstr
    rho_cstr = Mw_mono * C_assu_cstr  # g/m^3

    Cp_C2 = 42.9
    Cp_ass_cstr = Cp_C2 / Mw_mono

    U_heat_cstr = 400
    D_cstr = 0.05

    # ---- CSTR operating conditions -----------------------------
    mono_0 = 8000.0          # mol/m^3
    ini_0  = 50.0            # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0    = 200 + 273.15    # K

    tau_cstr = 5.0           # s

    t_end = 20.0             # s
    dt = 0.01                # s

    # ---- Model object for CSTR ---------------------------------
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
    )

    cstr_model.set_cstr_params(
        Tc_const=Tc_const_cstr,
        rho=rho_cstr,
        Cp_ass=Cp_ass_cstr,
        U_heat=U_heat_cstr,
        D=D_cstr,
    )

    # ---- Run CSTR simulation -----------------------------------
    cstr_res = cstr_model.run_cstr(
        mono_0=mono_0,
        ini_0=ini_0,
        CTA_0=CTA_0,
        T_0=T_0,
        tau=tau_cstr,
        t_end=t_end,
        dt=dt,
    )

    # ---- Print final result ------------------------------------
    print("\n========== CSTR result ==========")
    print(f"tau        = {cstr_res['tau']:.4f} s")
    print(f"Mn_final   = {cstr_res['Mn_final']:.4e} g/mol")
    print(f"Mw_final   = {cstr_res['Mw_final']:.4e} g/mol")
    print(f"PDI_final  = {cstr_res['PDI_final']:.4f}")
    print(f"T_final    = {cstr_res['T_final']:.2f} K")
    print(f"mono_final = {cstr_res['mono_final']:.4e} mol/m^3")
    print(f"ini_final  = {cstr_res['ini_final']:.4e} mol/m^3")

    # ---- Plot CSTR result --------------------------------------
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


# %%
# PFR simulation
if __name__ == "__main__":
    # ============================================================
    # Common kinetic parameters
    # ============================================================
    kd_test = 0.8
    kp_test = 15
    ktc_test = 0.02
    ktd_test = 0.02
    ktrm_test = 15E-3
    ktrp_test = 10E-3
    kca_test = 5E-3
    f_test = 0.8  # Initiator efficiency

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

    # ============================================================
    # Common physical parameters
    # ============================================================
    dH_p = -100 * 1000  # J/mol
    Mw_mono = 28.05     # g/mol
    R_gas = 8.3145      # J/mol/K

    # ============================================================
    # PFR parameters
    # ============================================================
    mono_0 = 8000.0          # mol/m^3
    ini_0 = 50.0             # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0 = 200 + 273.15       # K
    Tc_in = 200 + 273.15     # K

    t_end = 20.0             # s

    N = 200                  # number of axial grid points
    h_r = 0.5                # reactor heat transfer term
    h_j = 0.5                # jacket heat transfer term

    # PFR geometry / flow parameters
    L_pfr = 10.0             # m
    D_pfr = 0.05             # m
    D_j_pfr = 0.08           # m
    v_pfr = 0.5              # m/s
    v_c_pfr = 0.5            # m/s

    # PFR physical parameters
    P_assu_pfr = 3000 * 1E5
    T_assu_pfr = 150 + 273.15
    C_assu_pfr = P_assu_pfr / R_gas / T_assu_pfr
    rho_pfr = Mw_mono * C_assu_pfr

    Cp_C2 = 42.9
    Cp_pfr = Cp_C2 / Mw_mono

    rho_c_pfr = 1000.0       # coolant density
    Cp_c_pfr = 4180.0        # coolant heat capacity

    # PFR Arrhenius parameters
    # 현재는 Batch/CSTR와 같은 속도상수를 쓰기 위해 Ea = 0으로 설정
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

    # PFR pressure correction
    P_ref_bar = 3000.0
    DV_kp = 0.0
    # ---- PFR operating conditions ------------------------------
    mono_0 = 8000.0          # mol/m^3
    ini_0  = 50.0            # mol/m^3
    CTA_0 = 50.0             # mol/m^3
    T_0    = 200 + 273.15    # K
    Tc_in  = 200 + 273.15    # K

    t_end = 20.0             # s

    N = 200                  # number of axial grid points
    h_r = 0.5                # reactor heat transfer term
    h_j = 0.5                # jacket heat transfer term

    # ---- Model object for PFR ----------------------------------
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

    pfr_model.set_pfr_params(
        # ============================================================
        # PFR geometry / flow parameters
        # ============================================================
        L=L_pfr,           # Reactor length [m]
        D=D_pfr,           # Reactor inner diameter [m]
        D_j=D_j_pfr,       # Jacket inner diameter [m]
        v=v_pfr,           # Reactor-side superficial velocity [m/s]
        v_c=v_c_pfr,       # Coolant velocity [m/s]

        # ============================================================
        # PFR physical parameters
        # ============================================================
        rho_pfr=rho_pfr,       # Reactor fluid density [g/m^3]
        Cp_pfr=Cp_pfr,         # Reactor fluid heat capacity [J/g/K]
        rho_c=rho_c_pfr,       # Coolant density [kg/m^3]
        Cp_c=Cp_c_pfr,         # Coolant heat capacity [J/kg/K]

        # ============================================================
        # Arrhenius parameters
        # Ea = 0이므로 A 값이 실제 속도상수와 동일하게 사용됨
        # ============================================================
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

        # ============================================================
        # Pressure correction parameters
        # 현재 DV_kp = 0이므로 압력 보정은 적용되지 않음
        # ============================================================
        P_ref_bar=P_ref_bar,
        DV_kp=DV_kp,
    )

    # ---- Initial condition for PFR ------------------------------
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

    # ---- Build PFR ODE function --------------------------------
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

    # ---- Run PFR simulation -------------------------------------
    pfr_sol = solve_ivp(
        pfr_rhs,
        t_span=(0.0, t_end),
        y0=y0_pfr,
        method="BDF",
        atol=atol,
        rtol=1e-6,
        jac_sparsity=jac_sp,
    )

    # ---- Extract final axial profile ----------------------------
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

    eps = 1E-30

    Mn_pfr = pfr_model.Mw_mono * (lamb1_pfr + mu1_pfr + eps) / (lamb0_pfr + mu0_pfr + eps)
    Mw_pfr = pfr_model.Mw_mono * (lamb2_pfr + mu2_pfr + eps) / (lamb1_pfr + mu1_pfr + eps)
    PDI_pfr = Mw_pfr / Mn_pfr

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
        "CTA_final": CTA_pfr[-1],
        "ini_final": ini_pfr[-1],
    }

    # ---- Print final result -------------------------------------
    print("\n========== PFR result ==========")
    print(f"success     = {pfr_sol.success}")
    print(f"message     = {pfr_sol.message}")
    print(f"Mn_outlet   = {pfr_res['Mn_final']:.4e} g/mol")
    print(f"Mw_outlet   = {pfr_res['Mw_final']:.4e} g/mol")
    print(f"PDI_outlet  = {pfr_res['PDI_final']:.4f}")
    print(f"T_outlet    = {pfr_res['T_final']:.2f} K")
    print(f"mono_outlet = {pfr_res['mono_final']:.4e} mol/m^3")
    print(f"ini_outlet  = {pfr_res['ini_final']:.4e} mol/m^3")

    # ---- Plot PFR axial profiles --------------------------------
    plt.figure()
    plt.plot(pfr_res["z"], pfr_res["T"], label="PFR")
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