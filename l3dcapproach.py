import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fsolve, differential_evolution
from mpl_toolkits.mplot3d import Axes3D

def solve_l2point(mu):
    # quintic
    alpha = (mu/(3 * (1 - mu)))**(1/3)
    u0 = alpha * (1 + alpha/3 - alpha**2/9)
    def l2_eqn(u):
        return u**2 * (u**3 + 3*u**2 + 3*u) - mu*(u**4 + 2*u**3 + u**2 + 2*u + 1)
    u = fsolve(l2_eqn, x0=u0)
    return u[0]

def get_jacobian(state, mu):
    x, y, z, xdot, ydot, zdot = state[:]
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    Uxx = 1 - (1-mu)/r1**3 - mu/r2**3 +3*((x+mu)**2 * (1-mu)/r1**5 + (x + mu - 1)**2 * mu/r2**5)
    Uyy = 1 - (1-mu)/r1**3 - mu/r2**3 +3*y**2*((1-mu)/r1**5 + mu/r2**5)
    Uzz = -(1-mu)/r1**3 - mu/r2**3 + 3*z**2*((1-mu)/r1**5 + mu/r2**5)
    Uxy = Uyx = 3*y*((x+mu)*(1-mu)/r1**5 + (x + mu - 1)*mu/r2**5)
    Uxz = Uzx = 3*z*((x+mu)*(1-mu)/r1**5 + (x + mu - 1)*mu/r2**5)
    Uyz = Uzy = 3*y*z*((1-mu)/r1**5 + mu/r2**5)
    U_lap = np.array([[Uxx, Uxy, Uxz], [Uyx, Uyy, Uyz], [Uzx, Uzy, Uzz]])

    coriolis = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    A = np.zeros((6, 6))
    A[:3,:3] = np.zeros((3,3))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = U_lap
    A[3:6,3:6] = 2 * coriolis
    return A

def f(t, state, mu):
    x, y, z, xdot, ydot, zdot = state[:]
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    xddot = 2*ydot + x - ((1-mu)/r1**3)*(x + mu) + (mu/r2**3)*(1 - mu - x)
    yddot = -2*xdot + y - ((1-mu)/r1**3)*y - (mu/r2**3)*y
    zddot = - ((1-mu)/r1**3)*z - (mu/r2**3)*z

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])

def derivs(t, full_state, mu):
    state = full_state[:6]
    Phi = full_state[6:].reshape(6, 6)
    A = get_jacobian(state, mu)
    Phidot = A @ Phi

    return np.concatenate((f(t, state, mu), Phidot.flatten()))

def deriv(t, X, mu):
    x, y, z, xdot, ydot, zdot = X
    r1 = np.sqrt((x      + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1. + mu)**2 + y**2 + z**2)

    term_1 = x + 2. * ydot
    term_2 = -(1.-mu) * (x + mu) / r1**3
    term_3 =     -mu  * (x - 1. + mu) / r2**3
    xddot  = term_1 + term_2 + term_3

    term_1 = -2. * xdot
    term_2 = 1. - (1.-mu)/r1**3 - mu/r2**3 
    yddot  = term_1 + y * term_2

    term_1 = (1. - mu)/r1**3 + mu/r2**3
    zddot  = -z * term_1

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])


def jacobi_constant(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x+mu)**2   + y**2 + z**2)
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)
    v2 = vx**2 + vy**2 + vz**2
    U  = (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2
    return 2*U - v2

M_E = 5.972e24
M_S = 1.989e30
r12 = AU = 149_597_870.700
mu = M_E / (M_E + M_S)

u2 = solve_l2point(mu)

# differential correction Breakwell & Brown (1979)
def iteration(X0,T):
    Phi0 = np.eye(6)
    state0 = np.concatenate((X0, Phi0.flatten()), axis=0)

    def y_zero(t, s, mu):
        return s[1]
    y_zero.terminal = False
    y_zero.direction = -1 

    n_half=1200
    sol = solve_ivp(
        derivs,
        args=(mu,),
        t_span = (0.0, T),
        y0 = state0,
        method = 'DOP853',
        events = y_zero,
        t_eval= T * np.linspace(0.0, 1.0, 2*n_half+1),
        rtol = 1e-11,
        atol = 1e-12,
        dense_output = False)

    if not sol.t_events[0].size:
        print('No Y crossing')
        return None

    s_half = sol.y_events[0][0]
    X_half = s_half[:6]
    Phi = s_half[6:].reshape((6,6))
    f_half = f(0,X_half, mu)

    M = np.array([[Phi[3,0] - (f_half[3]/f_half[1])*Phi[1,0],
                    Phi[3,4] - (f_half[3]/f_half[1])*Phi[1,4]],
                    [Phi[5,0] - (f_half[5]/f_half[1])*Phi[1,0],
                    Phi[5,4] - (f_half[5]/f_half[1])*Phi[1,4]]])
    delta_free = - np.linalg.inv(M) @ np.array([X_half[3], X_half[5]]).T

    return delta_free, sol


def BB1979(z0, mu, x0, ydot0, r12, T):

    X0 = np.array([x0, 0, z0, 0, ydot0, 0])
 
    u2 = solve_l2point(mu)
    xL2 = u2 + 1 - mu
    #fig, axs = plt.subplots(3,2,figsize=(10,8),dpi=100,projection='3d')
    fig = plt.figure(figsize=(8,10),dpi=200,constrained_layout=True)
    axs = []
    for k in range(4):
        delta_free, sol = iteration(X0, T)
        X0 = X0 + np.array([delta_free[0], 0, 0, 0, delta_free[1], 0])
        r, c = k % 6, k // 6
        ax = fig.add_subplot(3, 2, k+1, projection='3d')#axs[r, c]

        ax.plot([xL2*r12], [0], [0], 'ok', markersize=8)
        x, y, z = sol.y[:3]
        im = ax.scatter(x * r12, y * r12, z * r12, c=sol.t/T)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=310, roll=0)
 
        ax.set_xlim(1.007*r12, 1.028*r12)
        ax.set_ylim(-0.025*r12, 0.005*r12)
        ax.set_zlim(-0.0015*r12,0.0013*r12)

        axs.append(ax)
    
    fig.colorbar(im, ax=axs, fraction=0.01)
    plt.show()

    return X0


# free parameter is z0
z0 = 9.09018724526800e-04

x0 = 1.00833444281154e+00
ydot0 = 9.93564036550147e-03
X0 = BB1979(z0, mu, x0, ydot0, r12, T=3.5)

