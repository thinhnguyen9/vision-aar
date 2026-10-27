import numpy as np
from math import sin, cos, tan, sqrt, atan2
import casadi as ca

def saturate(val, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, val))

# ----------------------------------------------------
# Rotation matrices and helper functions
# ----------------------------------------------------
def rotmat(phi, theta, psi):
    """
    Rotation matrix body-to-world using ZYX (yaw-pitch-roll) convention.
    """
    cphi, sphi = cos(phi),   sin(phi)
    cth,  sth  = cos(theta), sin(theta)
    cpsi, spsi = cos(psi),   sin(psi)
    # R = np.array([
    #     [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],
    #     [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],
    #     [-sth,     cth*sphi,                  cth*cphi]
    # ])
    Rx = np.array([[1, 0, 0],
                    [0, cphi, -sphi],
                    [0, sphi, cphi]])
    Ry = np.array([[cth, 0, sth],
                    [0, 1, 0],
                    [-sth, 0, cth]])
    Rz = np.array([[cpsi, -spsi, 0],
                    [spsi, cpsi, 0],
                    [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

def dR_dTheta(phi, theta, psi):
    """
    Partial derivatives of rotation matrix w.r.t. roll, pitch, yaw angles.
    """
    cphi, sphi = cos(phi),   sin(phi)
    cth,  sth  = cos(theta), sin(theta)
    cpsi, spsi = cos(psi),   sin(psi)

    Rx = np.array([[1, 0, 0],
                    [0, cphi, -sphi],
                    [0, sphi, cphi]])
    Ry = np.array([[cth, 0, sth],
                    [0, 1, 0],
                    [-sth, 0, cth]])
    Rz = np.array([[cpsi, -spsi, 0],
                    [spsi, cpsi, 0],
                    [0, 0, 1]])
    
    dRx_dphi = np.array([[0, 0, 0],
                            [0, -sphi, -cphi],
                            [0, cphi, -sphi]])
    dRy_dtheta = np.array([[-sth, 0, cth],
                            [0, 0, 0],
                            [-cth, 0, -sth]])
    dRz_dpsi = np.array([[-spsi, -cpsi, 0],
                            [cpsi, -spsi, 0],
                            [0, 0, 0]])
    
    dR_dphi = Rz @ Ry @ dRx_dphi
    dR_dtheta = Rz @ dRy_dtheta @ Rx
    dR_dpsi = dRz_dpsi @ Ry @ Rx
    return dR_dphi, dR_dtheta, dR_dpsi

def Smat(phi, theta):
    """
    omega = Smat(phi,theta) * [phi_dot, theta_dot, psi_dot]^T
    """
    cphi, sphi = cos(phi),   sin(phi)
    cth,  sth  = cos(theta), sin(theta)
    S = np.array([
        [1, 0,     -sth    ],
        [0, cphi,  sphi*cth],
        [0, -sphi, cphi*cth]
    ])
    return S

def dS_dTheta(phi, theta):
    """
    Partial derivatives of S w.r.t. roll and pitch angles.
    """
    cphi, sphi = cos(phi),   sin(phi)
    cth,  sth  = cos(theta), sin(theta)
    dS_dphi = np.array([
        [0, 0, 0],
        [0, -sphi, cphi*cth],
        [0, -cphi, -sphi*cth]
    ])
    dS_dtheta = np.array([
        [0, 0, -cth],
        [0, 0, -sphi*sth],
        [0, 0, -cphi*sth]
    ])
    dS_dpsi = np.zeros((3,3))
    return dS_dphi, dS_dtheta, dS_dpsi

def W(phi, theta):
    """
    Euler-angle rate transformation matrix. W = S^-1
    """
    cphi, sphi = cos(phi),   sin(phi)
    cth, sth   = cos(theta), sin(theta)
    tth        = tan(theta)
    W = np.array([
        [1, sphi*tth, cphi*tth],
        [0, cphi, -sphi],
        [0, sphi/cth, cphi/cth]
    ])
    return W

def skew_symmetric(v):
    """
    Creates a 3x3 skew-symmetric matrix from a 3D vector.
    """
    return np.array([
        [0,     -v[2], v[1] ],
        [v[2],  0,     -v[0]],
        [-v[1], v[0],  0    ]
    ])

# ----------------------------------------------------
# For CasADi
# ----------------------------------------------------
def casadi_rotmat(phi, theta, psi):
    """
    phi, theta, psi: casadi SX symbolic variables
    """
    cphi, sphi = ca.cos(phi),   ca.sin(phi)
    cth,  sth  = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi),   ca.sin(psi)
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi, cphi)
    )
    Ry = ca.vertcat(
        ca.horzcat(cth, 0, sth),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sth, 0, cth)
    )
    Rz = ca.vertcat(
        ca.horzcat(cpsi, -spsi, 0),
        ca.horzcat(spsi, cpsi, 0),
        ca.horzcat(0, 0, 1)
    )
    R = Rz @ Ry @ Rx
    return R

def casadi_Smat(phi, theta):
    """
    phi, theta: casadi SX symbolic variables
    """
    cphi, sphi = ca.cos(phi),   ca.sin(phi)
    cth,  sth  = ca.cos(theta), ca.sin(theta)
    S = ca.vertcat(
        ca.horzcat(1, 0,     -sth    ),
        ca.horzcat(0, cphi,  sphi*cth),
        ca.horzcat(0, -sphi, cphi*cth)
    )
    return S

def casadi_skew_symmetric(v):
    """
    v: casadi SX symbolic 3D vector (3x1)
    """
    return ca.vertcat(
        ca.horzcat(0,     -v[2], v[1] ),
        ca.horzcat(v[2],  0,     -v[0]),
        ca.horzcat(-v[1], v[0],  0    )
    )
