import numpy as np
import casadi as ca
from .dynamical_system import DynamicalSystem
from .utils import *


class AAR_Kinematics_v1(DynamicalSystem):
    """
    AAR Kinematics Model for MPC planner. Adopted from AAR_relative_distance_model_v2
    (2026-08-19) for CasADi. Only keep receiver aircraft's states; pdt & vdt are treated
    as external time-varying disturbances.

    States (7): x = [
        dx, dy, dz,         # relative probe-to-drogue distance (m) in receiver frame (NED)
        phi, theta, psi,    # receiver's attitudes (roll, pitch, yaw) (rad)
        v                   # receiver's total airspeed (m/s) in body frame
    ]

    Controls (3): u = [
        phi_dot,        # roll rate (rad/s)
        theta_dot,      # pitch rate (rad/s)
        v_dot           # airspeed rate (m/s^2)
    ]

    Disturbances (3): w = [
        vdt_x, vdt_y, vdt_z     # drogue's velocity in tanker frame (m/s)
    ]

    Measurements (7): y = [
        pdc_x, pdc_y, pdc_z,    # vision: drogue position in receiver's camera frame (m)
        phi, theta, psi, v      # from receiver's avionics
    ]
    """

    def __init__(
            self,
            tanker_vel_body=np.array([100., 0., 0.]),           # tanker velocity in body frame (m/s)
            tanker_attitudes=np.array([0., 0., 0.]),            # tanker attitudes (roll/pitch/yaw) (rad)
            receiver_probe_pos=np.array([7.2, 1.05, -0.04]),    # probe position (fwd/rght/dwn) in receiver frame (m)
            receiver_camera_pos=np.array([1.4439, 1.1196, -0.5051]),
            receiver_camera_attitudes=np.array([0., 0., 0.]),   # receiver camera attitudes (roll/pitch/yaw) (rad)
            gravitational_acceleration=9.81                     # (m/s^2)
        ):
        # Model dimensions
        self.Nx = 7
        self.Nu = 3
        self.Nw = 3
        self.Ny = 7

        # Physics
        self._g = gravitational_acceleration

        # Tanker aircraft parameters
        self._tanker_vel_body = tanker_vel_body
        self._tanker_attitudes = tanker_attitudes
        self._R_TW = rotmat(self._tanker_attitudes[0],
                            self._tanker_attitudes[1],
                            self._tanker_attitudes[2])  # Rotation matrix of tanker w.r.t. world frame
        # self._tanker_vel_world = self._R_TW @ self._tanker_vel_body  # Tanker velocity in world frame (m/s) (w/ assumptions!)

        # Receiver aircraft parameters
        # The receiver frame origin is where the `/position/` attribute in FlightGear refers to.
        self._receiver_probe_pos = receiver_probe_pos
        self._receiver_camera_pos = receiver_camera_pos
        self._receiver_camera_attitudes = receiver_camera_attitudes
        self._R_CR = rotmat(self._receiver_camera_attitudes[0],
                            self._receiver_camera_attitudes[1],
                            self._receiver_camera_attitudes[2])  # Rotation matrix of camera w.r.t. receiver body frame

        self._trim_pitch = np.deg2rad(4.1826)  # trim pitch angle (rad)
        
        # Define symbolic variables
        self._setup_casadi()
    
    # --------------------------- Required methods --------------------------- #
    def state_dot(self, current_state, current_input, current_disturbance):
        return self._f_continuous(current_state, current_input, current_disturbance).full().flatten()

    def state_next(self, current_state, current_input, current_disturbance, dt):
        return self._f_discrete(current_state, current_input, current_disturbance, dt).full().flatten()
    
    def linearize(self, nominal_state, nominal_input, nominal_disturbance):
        measurement_noise = np.zeros(self.Ny)
        A = self._A_func(nominal_state, nominal_input, nominal_disturbance).full()
        B = self._B_func(nominal_state, nominal_input, nominal_disturbance).full()
        G = self._G_func(nominal_state, nominal_input, nominal_disturbance).full()
        C = self._C_func(nominal_state, measurement_noise).full()
        return A, B, G, C

    def get_output(self, current_state, current_noise):
        return self._h_func(current_state, current_noise).full().flatten()

    # --------------------------- Internal methods --------------------------- #
    def _setup_casadi(self):
        """
        Setup symbolic dynamics using CasADi
        """
        # States
        d = ca.SX.sym('d', 3)
        phi = ca.SX.sym('phi')
        theta = ca.SX.sym('theta')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        self._X = ca.vertcat(d, phi, theta, psi, v)
        
        # Control inputs
        phi_dot = ca.SX.sym('phi_dot')
        theta_dot = ca.SX.sym('theta_dot')
        v_dot = ca.SX.sym('v_dot')
        self._U = ca.vertcat(phi_dot, theta_dot, v_dot)

        # Disturbances
        vdt = ca.SX.sym('vdt', 3)
        self._W = vdt

        # Psi dynamics
        psi_dot = self._g * ca.tan(phi) / v     # coordinated turn

        # Intermediate vectors and matrices
        Theta = ca.vertcat(phi, theta, psi)
        Theta_dot = ca.vertcat(phi_dot, theta_dot, psi_dot)
        R_RW = casadi_rotmat(phi, theta, psi)
        omega = casadi_Smat(phi, theta) @ Theta_dot
        omega_skew = casadi_skew_symmetric(omega)

        # Relative probe-to-drogue velocity
        d_dot = R_RW.T @ ca.DM(self._R_TW) @ (vdt + ca.DM(self._tanker_vel_body)) \
                - v * ca.DM([cos(self._trim_pitch), 0., sin(self._trim_pitch)]) \
                - omega_skew @ (d + ca.DM(self._receiver_probe_pos))
        
        # State derivative
        self._Xdot = ca.vertcat(d_dot, Theta_dot, v_dot)

        # ------------------------------------------------------------------------- #
        # Continuous dynamics function: Function(name, [inputs], [outputs])
        self._f_continuous = ca.Function(
            'f_continuous', 
            [self._X, self._U, self._W],
            [self._Xdot],
            ['x', 'u', 'w'], ['x_dot']
        )

        # ------------------------------------------------------------------------- #
        # Discrete dynamics using RK4 integration
        dt = ca.SX.sym('dt')
        k1 = self._f_continuous(self._X,             self._U, self._W)
        k2 = self._f_continuous(self._X + dt/2 * k1, self._U, self._W)
        k3 = self._f_continuous(self._X + dt/2 * k2, self._U, self._W)
        k4 = self._f_continuous(self._X + dt * k3,   self._U, self._W)
        Xnext = self._X + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self._f_discrete = ca.Function(
            'f_discrete',
            [self._X, self._U, self._W, dt],
            [Xnext],
            ['x', 'u', 'w', 'dt'], ['x_next']
        )

        # ------------------------------------------------------------------------- #
        # Output measurement function
        measurement_noise = ca.SX.sym('measurement_noise', self.Ny)
        pdc = ca.DM(self._R_CR).T @ (d + self._receiver_probe_pos - self._receiver_camera_pos)
        self._V = measurement_noise
        self._Y = ca.vertcat(pdc, Theta, v) + self._V
        self._h_func = ca.Function('h', [self._X, self._V], [self._Y], ['x', 'v'], ['y'])

        # ------------------------------------------------------------------------- #
        # Pre-compute Jacobian linearization
        A_sym = ca.jacobian(self._Xdot, self._X)
        B_sym = ca.jacobian(self._Xdot, self._U)
        G_sym = ca.jacobian(self._Xdot, self._W)
        C_sym = ca.jacobian(self._Y, self._X)
        self._A_func = ca.Function('A', [self._X, self._U, self._W], [A_sym], ['xstar', 'ustar', 'wstar'], ['A'])
        self._B_func = ca.Function('B', [self._X, self._U, self._W], [B_sym], ['xstar', 'ustar', 'wstar'], ['B'])
        self._G_func = ca.Function('G', [self._X, self._U, self._W], [G_sym], ['xstar', 'ustar', 'wstar'], ['G'])
        self._C_func = ca.Function('C', [self._X, self._V],          [C_sym], ['xstar', 'vstar'],          ['C'])
