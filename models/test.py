import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import AAR_Kinematics_v1
# from aar_dynamics_old import AAR_relative_distance_model_v2
import numpy as np


aar_new = AAR_Kinematics_v1()
# aar_old = AAR_relative_distance_model_v2()
xnom = np.array([10., 1., 1., 0.01, -.02, .1, 110.])

A_new, B_new, G_new, C_new = aar_new.linearize(
    nominal_state=xnom,
    nominal_input=np.zeros(aar_new.Nu),
    nominal_disturbance=np.zeros(aar_new.Nw)
)
# A_old, B_old, G_old, _ = aar_old.linearize(
#     state=np.hstack((np.zeros(6), xnom)),
#     control=np.zeros(aar_old.Nu)
# )
print("A (new):", A_new)
print("B (new):", B_new)
print("G (new):", G_new)
print("C (new):", C_new)
print("xdot(k):", aar_new.state_dot(xnom, np.zeros(aar_new.Nu), np.zeros(aar_new.Nw)))
print("x(k+1):", aar_new.state_next(xnom, np.zeros(aar_new.Nu), np.zeros(aar_new.Nw), 0.1))
print("y(k):", aar_new.get_output(xnom, np.zeros(aar_new.Ny)))
# print("A (old):", A_old)
# print("B (old):", B_old)
# print("G (old):", G_old)