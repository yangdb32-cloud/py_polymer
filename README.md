# PyPoly Reactor
PyPoly Reactor is a Python package for polymerization reactor modeling using the method of moments.
The package supports Batch, CSTR, and PFR polymerization simulations and calculates molecular-weight properties such as Mn, Mw, and PDI.

## Installation
It is recommended to install PyPoly Reactor in a Python virtual environment.
The virtual environment can be created in any folder selected by the user.

### 1. Create a project folder
Create a folder for your simulation project and open a terminal in that folder.
For example:

```bash
mkdir pypoly_simulation
cd pypoly_simulation
```

### 2. Create a virtual environment
Create a virtual environment with a name of your choice.
For example, to create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

You may use another name or path instead of `.venv`.
For example:

```bash
python -m venv my_pypoly_env
```

### 3. Activate the virtual environment
#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### macOS or Linux

```bash
source .venv/bin/activate
```

After activation, the virtual environment name should appear at the beginning of the terminal line.

For example:

```text
(.venv) C:\Users\User\pypoly_simulation>
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install PyPoly Reactor

Install the package inside the activated virtual environment:

```bash
python -m pip install pypoly-reactor
```

The package will be installed only in the currently activated virtual environment.

### 6. Open the project in VS Code

Open the current project folder in VS Code:

```bash
code .
```

Alternatively, open VS Code manually and select:

```text
File → Open Folder
```

Then select the project folder containing the virtual environment.

### 7. Select the virtual environment in VS Code

In VS Code:

1. Press `Ctrl + Shift + P`.
2. Search for `Python: Select Interpreter`.
3. Select the Python interpreter located in the virtual environment.

For a `.venv` environment on Windows, the interpreter is usually:

```text
.venv\Scripts\python.exe
```

For macOS or Linux, it is usually:

```text
.venv/bin/python
```

The selected Python environment is displayed in the bottom-right corner of VS Code.

If the environment does not appear in the interpreter list, select:

```text
Enter interpreter path
```

and manually locate the Python executable inside the virtual environment.

### 8. Verify the installation

Run the following command in the VS Code terminal:

```bash
python -m pip show pypoly-reactor
```

You can also verify the installation using Python:

```bash
python -c "import pypoly; print('PyPoly Reactor installed successfully')"
```

### 9. Create a simulation file

Create a Python file such as:

```text
Sim_Batch.py
```

Import the package as follows:

```python
from pypoly import PolyRxn
```

The simulation file will use the `pypoly-reactor` package installed in the virtual environment selected in VS Code.

## Changing the Virtual Environment

To use PyPoly Reactor in another virtual environment:

1. Create or activate the desired virtual environment.
2. Install `pypoly-reactor` in that environment.
3. Select that environment using `Python: Select Interpreter` in VS Code.

Each virtual environment manages its packages independently. Therefore, installing `pypoly-reactor` in one environment does not automatically install it in another environment.

## Deactivating the Virtual Environment

To leave the current virtual environment, run:

```bash
deactivate
```


"""
수정하기
"""
## Quick Start

PyPoly Reactor provides simulation functions for Batch, CSTR, and PFR polymerization reactors.

The package can be imported as follows:

```python
from pypoly import PolyRxn
```

The following kinetic and physical parameters are used in the examples.

```python
# Kinetic parameters
kd = 0.8
kp = 15.0
ktc = 0.02
ktd = 0.02
ktrm = 15e-3
ktrp = 10e-3
f = 0.8

# Physical parameters
dH_p = -100.0e3       # Heat of polymerization [J/mol]
Mw_mono = 28.05       # Monomer molecular weight [g/mol]
R_gas = 8.3145        # Gas constant [J/(mol K)]
```

Create a polymerization model:

```python
model = PolyRxn(
    kd=kd,
    kp=kp,
    ktc=ktc,
    ktd=ktd,
    ktrm=ktrm,
    ktrp=ktrp,
    f=f,
    dH_p=dH_p,
    Mw_mono=Mw_mono,
    R_gas=R_gas,
)
```

---

## Batch Reactor Example

A batch reactor has no continuous inlet or outlet during the reaction.

The concentrations and polymer properties change with time as polymerization proceeds.

Create a Python file named `Sim_Batch.py`:

```python
from pypoly import PolyRxn


# ============================================================
# 1. Kinetic parameters
# ============================================================
kd = 0.8
kp = 15.0
ktc = 0.02
ktd = 0.02
ktrm = 15e-3
ktrp = 10e-3
f = 0.8


# ============================================================
# 2. Physical parameters
# ============================================================
dH_p = -100.0e3
Mw_mono = 28.05
R_gas = 8.3145


# ============================================================
# 3. Create the polymerization model
# ============================================================
batch_model = PolyRxn(
    kd=kd,
    kp=kp,
    ktc=ktc,
    ktd=ktd,
    ktrm=ktrm,
    ktrp=ktrp,
    f=f,
    dH_p=dH_p,
    Mw_mono=Mw_mono,
    R_gas=R_gas,
)


# ============================================================
# 4. Initial conditions
# ============================================================
mono_0 = 8000.0       # Initial monomer concentration [mol/m^3]
ini_0 = 50.0          # Initial initiator concentration [mol/m^3]
CTA_0 = 0.0           # Initial chain-transfer-agent concentration
T_0 = 350.0           # Initial reactor temperature [K]

t_end = 100.0         # Final simulation time [s]
dt = 0.1              # Time interval [s]


# ============================================================
# 5. Run the Batch simulation
# ============================================================
batch_result = batch_model.run_batch(
    mono_0=mono_0,
    ini_0=ini_0,
    CTA_0=CTA_0,
    T_0=T_0,
    t_end=t_end,
    dt=dt,
)


# ============================================================
# 6. Check the result
# ============================================================
print(batch_result)
```

Run the simulation:

```bash
python Sim_Batch.py
```

The batch simulation calculates the time-dependent reactor conditions and polymer properties, including:

* monomer concentration
* initiator concentration
* reactor temperature
* monomer conversion
* number-average molecular weight, Mn
* weight-average molecular weight, Mw
* polydispersity index, PDI

A complete example is also available in:

```text
examples/Sim_Batch.py
```

---

## CSTR Example

A continuous stirred-tank reactor, or CSTR, continuously receives reactants and removes the reactor mixture.

Because the reactor is assumed to be perfectly mixed, the outlet conditions are identical to the conditions inside the reactor.

Create a Python file named `Sim_CSTR.py`:

```python
from pypoly import PolyRxn


# ============================================================
# 1. Kinetic parameters
# ============================================================
kd = 0.8
kp = 15.0
ktc = 0.02
ktd = 0.02
ktrm = 15e-3
ktrp = 10e-3
f = 0.8


# ============================================================
# 2. Physical parameters
# ============================================================
dH_p = -100.0e3
Mw_mono = 28.05
R_gas = 8.3145


# ============================================================
# 3. Create the polymerization model
# ============================================================
cstr_model = PolyRxn(
    kd=kd,
    kp=kp,
    ktc=ktc,
    ktd=ktd,
    ktrm=ktrm,
    ktrp=ktrp,
    f=f,
    dH_p=dH_p,
    Mw_mono=Mw_mono,
    R_gas=R_gas,
)


# ============================================================
# 4. Feed and operating conditions
# ============================================================
mono_0 = 8000.0       # Feed monomer concentration [mol/m^3]
ini_0 = 50.0          # Feed initiator concentration [mol/m^3]
CTA_0 = 0.0           # Feed chain-transfer-agent concentration
T_0 = 350.0           # Feed and initial reactor temperature [K]

tau = 20.0             # Residence time [s]
t_end = 200.0          # Final simulation time [s]
dt = 0.1               # Time interval [s]


# ============================================================
# 5. Run the CSTR simulation
# ============================================================
cstr_result = cstr_model.run_cstr(
    mono_0=mono_0,
    ini_0=ini_0,
    CTA_0=CTA_0,
    T_0=T_0,
    tau=tau,
    t_end=t_end,
    dt=dt,
)


# ============================================================
# 6. Check the result
# ============================================================
print(cstr_result)
```

Run the simulation:

```bash
python Sim_CSTR.py
```

The residence time is defined as:

```text
tau = reactor volume / volumetric flow rate
```

The CSTR result shows the transient response from the initial reactor condition to the steady state.

At steady state, the concentrations and polymer properties no longer change significantly with time.

A complete example is also available in:

```text
examples/Sim_CSTR.py
```

---

## PFR Example

A plug-flow reactor, or PFR, continuously transports the reaction mixture along the axial direction of the reactor.

The concentration, temperature, conversion, and polymer properties vary with reactor position.

Create a Python file named `Sim_PFR.py`:

```python
from pypoly import PolyRxn


# ============================================================
# 1. Kinetic parameters
# ============================================================
kd = 0.8
kp = 15.0
ktc = 0.02
ktd = 0.02
ktrm = 15e-3
ktrp = 10e-3
f = 0.8


# ============================================================
# 2. Physical parameters
# ============================================================
dH_p = -100.0e3
Mw_mono = 28.05
R_gas = 8.3145


# ============================================================
# 3. Create the polymerization model
# ============================================================
pfr_model = PolyRxn(
    kd=kd,
    kp=kp,
    ktc=ktc,
    ktd=ktd,
    ktrm=ktrm,
    ktrp=ktrp,
    f=f,
    dH_p=dH_p,
    Mw_mono=Mw_mono,
    R_gas=R_gas,
)


# ============================================================
# 4. PFR geometry and operating parameters
# ============================================================
pfr_model.set_pfr_params(
    L=10.0,            # Reactor length [m]
    D=0.10,            # Reactor inner diameter [m]
    D_j=0.12,          # Jacket diameter [m]
    v=0.50,            # Reactor-fluid velocity [m/s]
    v_c=0.50,          # Coolant velocity [m/s]

    rho_pfr=800.0,     # Reactor-fluid density [kg/m^3]
    Cp_pfr=2000.0,     # Reactor-fluid heat capacity [J/(kg K)]
    rho_c=1000.0,      # Coolant density [kg/m^3]
    Cp_c=4180.0,       # Coolant heat capacity [J/(kg K)]

    A_kd=kd,
    Ea_kd=0.0,

    A_kp=kp,
    Ea_kp=0.0,

    A_ktc=ktc,
    Ea_ktc=0.0,

    A_ktd=ktd,
    Ea_ktd=0.0,

    A_ktrm=ktrm,
    Ea_ktrm=0.0,

    A_ktrp=ktrp,
    Ea_ktrp=0.0,
)


# ============================================================
# 5. PFR inlet conditions
# ============================================================
mono_0 = 8000.0       # Inlet monomer concentration [mol/m^3]
ini_0 = 50.0          # Inlet initiator concentration [mol/m^3]
CTA_0 = 0.0           # Inlet chain-transfer-agent concentration
T_0 = 350.0           # Inlet reactor-fluid temperature [K]
T_c0 = 300.0          # Inlet coolant temperature [K]

t_end = 20.0          # Final simulation time or residence time [s]
dt = 0.01             # Integration interval


# ============================================================
# 6. Run the PFR simulation
# ============================================================
pfr_result = pfr_model.run_pfr(
    mono_0=mono_0,
    ini_0=ini_0,
    CTA_0=CTA_0,
    T_0=T_0,
    T_c0=T_c0,
    t_end=t_end,
    dt=dt,
)


# ============================================================
# 7. Check the result
# ============================================================
print(pfr_result)
```

Run the simulation:

```bash
python Sim_PFR.py
```

The PFR simulation calculates the variation of the following quantities along the reactor:

* monomer concentration
* initiator concentration
* reactor-fluid temperature
* coolant temperature
* monomer conversion
* Mn
* Mw
* PDI

The last calculated point represents the reactor outlet condition.

A complete example is also available in:

```text
examples/Sim_PFR.py
```

---

## Example Directory

The package repository contains complete simulation examples:

```text
examples/
├── Sim_Batch.py
├── Sim_CSTR.py
└── Sim_PFR.py
```

After downloading or cloning the repository, the examples can be executed from the project root:

```bash
python examples/Sim_Batch.py
python examples/Sim_CSTR.py
python examples/Sim_PFR.py
```

When the package is installed from PyPI, users may create their own Python files and copy the corresponding example code.

---

## Notes

The numerical values used in these examples are demonstration values.

Users should replace the following parameters with values appropriate for their own polymerization system:

* reaction-rate constants
* activation energies
* feed concentrations
* reactor dimensions
* residence time
* heat-transfer properties
* initial and coolant temperatures

Always check that the units of all input parameters are consistent.
