import numpy as np
from sympy import Symbol, Eq, solve
from scipy.optimize import fsolve

from models.CALfuncs import jac_f

# %% 参数设置-------------------------------------------
base_S = 2000000.
base_U = 690.
base_I = base_S / base_U
base_Z = base_U / base_I
kp1 = 1e-5 * base_U
ki1 = 0.06 * base_U
kp3 = 0.01 * base_Z
ki3 = 1. * base_Z
kp2 = 0.01 / base_Z
ki2 = 5 / base_Z
kp4 = 0.01 / base_Z
ki4 = 5. / base_Z
kp5 = 40.
ki5 = 10000.

Udcref = 2.1
udc = 2.1
Cdc = 0.02
Xf = 0.08
Rf = 0.008
Idref = 0.92
Iqref = 999.
# 系统参数
f_0 = 50.
omega0 = 2 * np.pi * f_0
RL = 0.08
XL = 0.8
Utref = 1.
C = 0.05
Rc = 0.00
U_g = 1.
# Pref = 0.85
# Qref = 0

Z = np.sqrt(RL ** 2 + XL ** 2)
Zc = complex(RL, XL)
thetaz = np.arctan(XL / RL)


def MDL_O13(y, t=0):
    # VSC变量 - ---------------------------------------------
    x_P, x_ut, theta_pll, x_pll, x_id, x_iq, u_vsc_x, u_vsc_y, i_vsc_x, i_vsc_y, iLx, iLy = y

    # Time simulation set
    # global Pref
    # if t < 3:
    #     Pref = 0.85
    # elif t < 103:
    #     Pref = 0.85 + 0.1 * (t - 3) / 100
    # else:
    #     Pref = 0.95

    # VSC中间变量--------------------------------------------
    # XY到dq变换
    i_vsc_d = i_vsc_x * np.cos(theta_pll) + i_vsc_y * np.sin(theta_pll)
    i_vsc_q = i_vsc_y * np.cos(theta_pll) - i_vsc_x * np.sin(theta_pll)
    u_vsc_d = u_vsc_x * np.cos(theta_pll) + u_vsc_y * np.sin(theta_pll)
    u_vsc_q = u_vsc_y * np.cos(theta_pll) - u_vsc_x * np.sin(theta_pll)
    Uvsc = complex(u_vsc_x, u_vsc_y)
    ut = abs(Uvsc)
    x_vsc_q = u_vsc_q / abs(Uvsc)
    # 功率
    Pvsc = u_vsc_x * i_vsc_x + u_vsc_y * i_vsc_y
    Qvsc = u_vsc_y * i_vsc_x - u_vsc_x * i_vsc_y
    # 中间变量
    idref = kp1 * (udc - Udcref) + ki1 * x_P
    iqref = kp3 * (ut - Utref) + ki3 * x_ut
    E_vsc_d = kp2 * (idref - i_vsc_d) + ki2 * x_id + (u_vsc_d - Xf * i_vsc_q)
    E_vsc_q = kp4 * (iqref - i_vsc_q) + ki4 * x_iq + (u_vsc_q + Xf * i_vsc_d)
    # dq到XY变换
    E_vsc_x = E_vsc_d * np.cos(theta_pll) - E_vsc_q * np.sin(theta_pll)
    E_vsc_y = E_vsc_q * np.cos(theta_pll) + E_vsc_d * np.sin(theta_pll)

    # 微分量 - -----------------------------------------
    dtheta_pll = kp5 * u_vsc_q + ki5 * x_pll
    ucy = u_vsc_y - Rc * (i_vsc_y - iLy)
    ucx = u_vsc_x - Rc * (i_vsc_x - iLx)
    di_vsc_x = (-Rf * i_vsc_x + Xf * i_vsc_y + E_vsc_x - u_vsc_x) * omega0 / Xf
    di_vsc_y = (-Rf * i_vsc_y - Xf * i_vsc_x + E_vsc_y - u_vsc_y) * omega0 / Xf
    diLx = (-RL * iLx + XL * iLy + u_vsc_x - U_g) * omega0 / XL
    diLy = (-RL * iLy - XL * iLx + u_vsc_y) * omega0 / XL

    du_vsc_x = Rc * (di_vsc_x - diLx) + (i_vsc_x - iLx + C * ucy) * omega0 / C
    du_vsc_y = Rc * (di_vsc_y - diLy) + (i_vsc_y - iLy - C * ucx) * omega0 / C

    # 微分方程（代数方程）-----------------------------
    # 功率外环
    dy = np.zeros(12, dtype=float)
    dy[0] = Pref-Pvsc
    dy[1] = ut-Utref
    # PLL环
    dy[2] = dtheta_pll
    dy[3] = u_vsc_q
    # 电流环
    dy[4] = idref - i_vsc_d
    dy[5] = iqref - i_vsc_q
    # 电路动态
    dy[6] = du_vsc_x
    dy[7] = du_vsc_y
    dy[8] = di_vsc_x
    dy[9] = di_vsc_y
    dy[10] = diLx
    dy[11] = diLy
    return dy


# %%计算平衡点-----------------------------------------------
Prange = np.arange(0, 1.3, 0.01)
eigv_real = np.zeros([Prange.size, 12])
eigv_imag = np.zeros([Prange.size, 12])

iLx = Symbol('iLx')
iLy = Symbol('iLy')
ix = Symbol('ix')
iy = Symbol('iy')
ux = Symbol('ux')
uy = Symbol('uy')

for idx, Pidx in enumerate(Prange):
    Pref = Pidx

    eq1 = Eq(ux * ix + uy * iy - Pref, 0)
    eq2 = Eq(ux ** 2 + uy ** 2 - Utref, 0)
    eq3 = Eq(ux - U_g - RL * iLx + XL * iLy, 0)
    eq4 = Eq(uy - RL * iLy - XL * iLx, 0)
    eq5 = Eq(ix - iLx + uy * C, 0)
    eq6 = Eq(iy - iLy - ux * C, 0)
    S = solve([eq1, eq2, eq3, eq4, eq5, eq6], [ux, uy, iLx, iLy, ix, iy])
    # print(S, type(S))
    temp1 = np.angle(complex(S[0][0], S[0][1]), deg=False)
    if temp1 <= np.pi - np.arctan(XL / RL) & temp1 >= 0:
        ep_ux = float(S[0][0])
        ep_uy = float(S[0][1])
        ep_ix = float(S[0][2])
        ep_iy = float(S[0][3])
        ep_ilx = float(S[0][4])
        ep_ily = float(S[0][5])

    else:
        ep_ux = float(S[1][0])
        ep_uy = float(S[1][1])
        ep_ix = float(S[1][2])
        ep_iy = float(S[1][3])
        ep_ilx = float(S[1][4])
        ep_ily = float(S[1][5])

    ep_uc = complex(ep_ux, ep_uy)
    theta_pll = np.angle(ep_uc)

    y_ep0 = np.hstack(([0., 0., theta_pll, 0., 0., 0.], [ep_ux, ep_uy, ep_ix, ep_iy, ep_ilx, ep_ily]))
    print('电压', ep_uc)

    y_ep = fsolve(MDL_O13, y_ep0)
    print('平衡点状态量', y_ep)

    J_A = jac_f(MDL_O13, y_ep)
    EE, VV = np.linalg.eig(J_A)
    print(EE)
    eigv_real[idx][:] = np.real(EE)
    eigv_imag[idx][:] = np.imag(EE)
    if any(np.real(EE) > 0):
        print('UNSTABLE equibrium point!')