import casadi as ca
import numpy as np
from models import AAR_Kinematics_v1
from .utils import lqr


class MPC_Planner_v1():
    """
    Standard nonlinear MPC planner with CasADi + IPOPT solver.
    Decision variables: Z := [x(0)...x(H), u(0)...u(H-1)].
    Cost: J := sum (x - xref)^T Q (x - xref) + (u - uref)^T R (u - uref).
    Constraints:
        x(0) = x0,
        x(k+1) = f(x(k), u(k), w(k)),   k=0,...,H-1,
        xmin(k) <= x(k) <= xmax(k),     k=0,...,H,
        umin(k) <= u(k) <= umax(k),     k=0,...,H-1,
    where w(k)'s are external time-varying parameters (e.g., predictions from LSTM).
    """
    def __init__(
            self,
            model: AAR_Kinematics_v1,
            horizon: int,
            dt: float,
            Q: np.ndarray,      # (H+1, Nx, Nx)
            R: np.ndarray,      # (H,   Nu, Nu)
            xmin: np.ndarray,   # (H+1, Nx)
            xmax: np.ndarray,   # (H+1, Nx)
            umin: np.ndarray,   # (H,   Nu)
            umax: np.ndarray,   # (H,   Nu)
            lqr_terminal_cost=True
        ):
        self.model = model
        self.H = horizon
        self.dt = dt
        self.Nx = model.Nx
        self.Nu = model.Nu
        self.Nw = model.Nw
        self.lqr_terminal_cost = lqr_terminal_cost

        # Q, R are sequences of weights for the entire horizon
        # if lqr_terminal_cost=True, the terminal Q[H] will be replaced with LQR cost
        self.set_cost_weights(Q, R)

        # xmin/xmax/umin/umax are sequences of bounds for the entire horizon
        self.set_state_bounds(xmin, xmax)
        self.set_control_bounds(umin, umax)

        # Build the MPC NLP once
        self._Z0 = None     # previous solution, (Nx*(H+1) + Nu*H, )
        self._build_nlp()


    # --------------------------- Update class attributes --------------------------- #
    def set_cost_weights(self, Q: np.ndarray | None=None, R: np.ndarray | None=None):
        if Q is not None:
            if Q.shape != (self.H+1, self.Nx, self.Nx):
                raise ValueError(f"Q must have shape {(self.H+1, self.Nx, self.Nx)}, but got {Q.shape}")
            self.Q = Q
        if R is not None:
            if R.shape != (self.H, self.Nu, self.Nu):
                raise ValueError(f"R must have shape {(self.H, self.Nu, self.Nu)}, but got {R.shape}")
            self.R = R

    def set_state_bounds(self, lower_bounds: np.ndarray | None=None, upper_bounds: np.ndarray | None=None):
        if lower_bounds is not None:
            if lower_bounds.shape != (self.H+1, self.Nx):
                raise ValueError(f"State lower bounds must have shape {(self.H+1, self.Nx)}, but got {lower_bounds.shape}")
            self.xmin = lower_bounds
        if upper_bounds is not None:
            if upper_bounds.shape != (self.H+1, self.Nx):
                raise ValueError(f"State upper bounds must have shape {(self.H+1, self.Nx)}, but got {upper_bounds.shape}")
            self.xmax = upper_bounds
        
    def set_control_bounds(self, lower_bounds: np.ndarray | None=None, upper_bounds: np.ndarray | None=None):
        if lower_bounds is not None:
            if lower_bounds.shape != (self.H, self.Nu):
                raise ValueError(f"Control lower bounds must have shape {(self.H, self.Nu)}, but got {lower_bounds.shape}")
            self.umin = lower_bounds
        if upper_bounds is not None:
            if upper_bounds.shape != (self.H, self.Nu):
                raise ValueError(f"Control upper bounds must have shape {(self.H, self.Nu)}, but got {upper_bounds.shape}")
            self.umax = upper_bounds


    # --------------------------- Main methods --------------------------- #
    def extract_solution(self, solution):
        solution = np.array(solution).flatten()
        X = solution[ : (self.H + 1)*self.Nx].reshape((self.H + 1, self.Nx))
        U = solution[(self.H + 1)*self.Nx : ].reshape((self.H, self.Nu))
        return np.array(X), np.array(U)
    
    def compute_control(
            self,
            curr_state: np.ndarray,             # (Nx,)
            ref_states: np.ndarray,             # (H+1, Nx)
            ref_controls: np.ndarray,           # (H,   Nu)
            ref_disturbances: np.ndarray,       # (H,   Nw)
            Q: np.ndarray | None = None,        # (H+1, Nx, Nx), only update if needed
            R: np.ndarray | None = None,        # (H,   Nu, Nu), only update if needed
            xmin: np.ndarray | None = None,     # (H+1, Nx),     only update if needed
            xmax: np.ndarray | None = None,     # (H+1, Nx),     only update if needed
            umin: np.ndarray | None = None,     # (H,   Nu),     only update if needed
            umax: np.ndarray | None = None,     # (H,   Nu),     only update if needed
        ):
        """
        Main wrapper, call once every MPC step. Do:
            1. Update NLP with time-varying parameters:
                - Required: x0, Xref, Uref, W
                - Ad-hoc: Q, R, xmin, xmax, umin, umax
            2. Set terminal weight with LQR (if lqr_terminal_cost=True).
            3. Solve the NLP with IPOPT (warm-started from 'self._Z0').
            4. Extract and return X_opt, U_opt from the NLP solution.
        """
        # Update internal attributes at each MPC step (if not None)
        self.set_cost_weights(Q, R)
        self.set_state_bounds(xmin, xmax)
        self.set_control_bounds(umin, umax)
        self._set_lqr_terminal_cost(ref_states[-1], ref_controls[-1], ref_disturbances[-1])

        # Get NLP with new parameters
        nlp_eval = self._evaluate_nlp(
            x0=curr_state,
            Xref=ref_states,
            Uref=ref_controls,
            W=ref_disturbances
        )

        # Solve NLP
        try:
            sol = self.solver(
                x0=nlp_eval["z0"],
                lbx=nlp_eval["lbz"],
                ubx=nlp_eval["ubz"],
                lbg=nlp_eval["lbg"],
                ubg=nlp_eval["ubg"],
                p=nlp_eval["p_val"]
            )
            
            # Extract solution
            Z_opt = sol["x"].full().flatten()
            J_opt = sol["f"].full().flatten()[0]
            X_opt, U_opt = self.extract_solution(Z_opt)
            
            # Get solver stats
            stats = self.solver.stats()
            
            result = {
                "success": stats["success"],
                "X_opt": X_opt,
                "U_opt": U_opt,
                "cost": J_opt,
                "stats": stats
            }
            
            self.solver_stats = stats
            self._Z0 = Z_opt
            
            return result
            
        except Exception as e:
            print(f"Solver failed with error: {e}")
            result = {
                "success": False,
                "X_opt": None,
                "U_opt": None,
                "cost": None,
                "stats": None
            }
            return result


    # --------------------------- Internal methods --------------------------- #
    def _set_lqr_terminal_cost(self, terminal_x: np.ndarray,
                               terminal_u: np.ndarray, terminal_w: np.ndarray):
        if self.lqr_terminal_cost:
            A, B, _, _ = self.model.linearize(terminal_x, terminal_u, terminal_w)
            _, P, _ = lqr(A, B, self.Q[-1], self.R[-1])
            self.Q[-1] = P
            
    def _build_nlp(self):
        """
        Build the CasADi MPC nonlinear program 'self.mpc_nlp' and initialize 'self.solver'.
        """
        # Decision variables
        X = ca.SX.sym('X', self.Nx, self.H + 1)
        U = ca.SX.sym('U', self.Nu, self.H)
        Z = ca.vertcat(
            ca.reshape(X, -1, 1),   # stacks columns of a matrix
            ca.reshape(U, -1, 1)
        )

        # Parameters (time-varying references/ disturbances/ etc.)
        x0 = ca.SX.sym('x0', self.Nx)
        Xref = ca.SX.sym('Xref', self.Nx, self.H + 1)
        Uref = ca.SX.sym('Uref', self.Nu, self.H)
        W = ca.SX.sym('W', self.Nw, self.H)
        Q = ca.SX.sym('Q', self.Nx, self.Nx, self.H + 1)
        R = ca.SX.sym('R', self.Nu, self.Nu, self.H)
        P = ca.vertcat(
            x0,
            ca.reshape(Xref, -1, 1),
            ca.reshape(Uref, -1, 1),
            ca.reshape(W, -1, 1)
        )
        for k in range(self.H + 1):     # casadi cannot handle 3D array
            P = ca.vertcat(P, ca.reshape(Q[k], -1, 1))
        for k in range(self.H):
            P = ca.vertcat(P, ca.reshape(R[k], -1, 1))
        
        # Build objective and constraints
        J = 0
        g = []
        g.append(X[:,0] - x0)
        for k in range(self.H):
            X_err = X[:,k] - Xref[:,k]
            U_err = U[:,k] - Uref[:,k]
            J += X_err.T @ Q[k] @ X_err + U_err.T @ R[k] @ U_err
            x_next = self.model._f_discrete(X[:,k], U[:,k], W[:,k], self.dt)
            g.append(X[:,k+1] - x_next)
        X_err = X[:,self.H] - Xref[:,self.H]
        J += X_err.T @ Q[self.H] @ X_err

        # Build NLP
        self.mpc_nlp = {
            "x": Z,
            "f": J,
            "g": ca.vertcat(*g),
            "p": P
        }

        # Setup solver
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': False,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'ipopt.tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-9,
            'ipopt.warm_start_mult_bound_push': 1e-9,
            'ipopt.mu_init': 1e-5,
        }
        self.solver = ca.nlpsol("mpc_solver", "ipopt", self.mpc_nlp, opts)

    def _evaluate_nlp(
            self,
            x0: np.ndarray,     # (Nx,)
            Xref: np.ndarray,   # (H+1, Nx)
            Uref: np.ndarray,   # (H, Nu)
            W: np.ndarray,      # (H, Nw)
        ):
        """
        Evaluate new parameters for the NLP.
        """
        if self._Z0 is not None:
            X, U = self.extract_solution(self._Z0)
            z0 = ca.vertcat(    # use previous solution with repeated last values
                X[1:].flatten(),
                X[-1:].flatten(),
                U[1:].flatten(),
                U[-1:].flatten()
            )
        else:
            z0 = ca.vertcat(
                Xref.flatten(),
                Uref.flatten()
            )
        # numpy.flatten() stacks rows: [[x1,y1],[x2,y2],...] -> [x1,y1,x2,y2,...]
        p_val = ca.vertcat(
            x0,
            Xref.flatten(),
            Uref.flatten(),
            W.flatten(),
            self.Q.flatten(),
            self.R.flatten()
        )
        # self.xmin[0], self.xmax[0] = x0, x0     # initial state must equal x0 exactly
        # Not needed, but we assume [xmin[0], xmax[0]] contains x0!!
        lbz = ca.DM(np.hstack((self.xmin.flatten(), self.umin.flatten())))
        ubz = ca.DM(np.hstack((self.xmax.flatten(), self.umax.flatten())))
        lbg = ca.DM.zeros(self.Nx * (self.H + 1))
        ubg = ca.DM.zeros(self.Nx * (self.H + 1))
        return {
            "z0": z0,
            "lbz": lbz,
            "ubz": ubz,
            "lbg": lbg,
            "ubg": ubg,
            "p_val": p_val
        }
    
    
    
    

    