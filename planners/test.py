import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models import AAR_Kinematics_v1
from planners import MPC_Planner_v1
from utils import plot_results

# System & references
model = AAR_Kinematics_v1()
Nx, Nu, Nw, Ny = model.Nx, model.Nu, model.Nw, model.Ny
x0 = np.array([10., 1., -1.,
               0., model._trim_pitch, 0.,
               100.])
xref = np.array([5., 0., 0.,
                 0., model._trim_pitch, 0.,
                 100.])

# MPC planner
H = 100
dt = 0.1
Q = np.eye(Nx)
R = np.eye(Nu)
# State bounds must contain x0 (and ideally xref)
xmin = np.array([0.5, -2.0, -2.0,
                 np.deg2rad(-10), np.deg2rad(0), np.deg2rad(-10),
                 80.0])
xmax = np.array([15.0, 2.0, 2.0,
                 np.deg2rad(10), np.deg2rad(8), np.deg2rad(10),
                 120.0])
umin = np.array([np.deg2rad(-5), np.deg2rad(-5), -3.])
umax = np.array([np.deg2rad(5),  np.deg2rad(5),   3.])

Xref = np.tile(xref, (H+1, 1))
Uref = np.zeros((H, Nu))
W_sin = np.tile(np.sin(np.arange(H)).reshape(-1, 1), (1, 3))
W_const = np.zeros((H, Nw))

mpc = MPC_Planner_v1(
    model=model,
    horizon=H,
    dt=dt,
    Q=np.tile(Q, (H+1, 1, 1)),      # (H+1, Nx, Nx)
    R=np.tile(R, (H, 1, 1)),        # (H,   Nu, Nu)
    xmin=np.tile(xmin, (H+1, 1)),   # (H+1, Nx)
    xmax=np.tile(xmax, (H+1, 1)),   # (H+1, Nx)
    umin=np.tile(umin, (H, 1)),     # (H,   Nu)
    umax=np.tile(umax, (H, 1)),     # (H,   Nu)
    lqr_terminal_cost=True
)

# Compute open-loop control
mpc_res = mpc.compute_control(
    curr_state=x0,              # (Nx,)
    ref_states=Xref,            # (H+1, Nx)
    ref_controls=Uref,          # (H,   Nu)
    ref_disturbances=W_const,   # (H,   Nw)
)
print(f"[test_mpc] MPC solver success: {mpc_res['success']}")
print(f"[test_mpc] MPC cost: {mpc_res['cost']:.4f}")

# Apply open-loop control on real system
Uvec = mpc_res['U_opt']
Xvec = np.zeros((H+1, Nx))
Xvec[0] = x0
for k in range(H):
    Xvec[k+1] = model.state_next(Xvec[k], Uvec[k], W_const[k], dt)

results = {
    "t"             : np.arange(H+1) * dt,
    "xref"          : Xref[:, 0],
    "yref"          : Xref[:, 1],
    "zref"          : Xref[:, 2],
    "phiref"        : Xref[:, 3],
    "thetaref"      : Xref[:, 4],
    "psiref"        : Xref[:, 5],
    "vref"          : Xref[:, 6],
    "x"             : Xvec[:, 0],
    "y"             : Xvec[:, 1],
    "z"             : Xvec[:, 2],
    "phi"           : Xvec[:, 3],
    "theta"         : Xvec[:, 4],
    "psi"           : Xvec[:, 5],
    "v"             : Xvec[:, 6],
    "phidotref"     : Uref[:, 0],
    "thetadotref"   : Uref[:, 1],
    "vdotref"       : Uref[:, 2],
    "phidot"        : Uvec[:, 0],
    "thetadot"      : Uvec[:, 1],
    "vdot"          : Uvec[:, 2],
}
plot_results(results, "planners/images/mpc_sim.png")
