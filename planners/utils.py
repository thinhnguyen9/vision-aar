import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

def lqr(A, B, Q, R):
    """
    Solves the continuous-time LQR problem.
    xdot = A x + B u
    Cost = integral (x^T Q x + u^T R u) dt
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    eigvals = np.linalg.eigvals(A - B @ K)
    return K, P, eigvals

def plot_results(results, png_path="planners/images/mpc_sim.png"):
    plot_order = (
        (0, ['xref', 'x'],            ['k--', 'k-'],       1.0,        ['x (m)', None]),
        (1, ['yref', 'y'],            ['k--', 'k-'],       1.0,        ['y (m)', None]),
        (2, ['zref', 'z'],            ['k--', 'k-'],       1.0,        ['z (m)', None]),
        (3, ['vdot', 'v'],            ['r--', 'k-'],       1.0,        ['v (m/s)', 'Time (s)']),
        (4, ['phidot', 'phi', 'psi'], ['r--', 'k-', 'b-'], 180./np.pi, ['roll, yaw (deg)', 'Time (s)']),
        (5, ['thetadot', 'theta'],    ['r--', 'k-'],       180./np.pi, ['pitch (deg)', 'Time (s)']),
    )
    plt.figure(figsize=(10, 6))
    for (idx, labels, styles, scaling, axlabels) in plot_order:
        plt.subplot(2, 3, idx+1)
        for i in range(len(labels)):
            N = min(len(results['t']), len(results[labels[i]]))
            plt.plot(
                results['t'][:N],
                results[labels[i]][:N]*scaling,
                styles[i], lw=1.3, label=labels[i]
            )
        plt.grid()
        plt.legend()
        plt.ylabel(axlabels[0])
        plt.xlabel(axlabels[1])

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"[plot_results] Saved plot to {png_path}")