import numpy as np
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fsolve, differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_l2point(mu):
    # quintic
    alpha = (mu/(3 * (1 - mu)))**(1/3)
    u0 = alpha * (1 + alpha/3 - alpha**2/9)
    def l2_eqn(u):
        return u**2 * (u**3 + 3*u**2 + 3*u) - mu*(u**4 + 2*u**3 + u**2 + 2*u + 1)
    u = fsolve(l2_eqn, x0=u0)
    return u[0]

def solve_l1point(mu):
    # quintic
    alpha = (mu/(3 * (1 - mu)))**(1/3)
    u0 = alpha * (1 - alpha/3 - alpha**2/9)
    def l1_eqn(u):
        return u**2 * (u**3 + 3*u**2 + 3*u) - mu*(u**4 + 2*u**3 - u**2 - 2*u - 1)
    u = fsolve(l1_eqn, x0=u0)
    return u[0]

def f(t, state, mu):
    x, y, z, xdot, ydot, zdot = state[:]
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    xddot = 2*ydot + x - ((1-mu)/r1**3)*(x + mu) + (mu/r2**3)*(1 - mu - x)
    yddot = -2*xdot + y - ((1-mu)/r1**3)*y - (mu/r2**3)*y
    zddot = - ((1-mu)/r1**3)*z - (mu/r2**3)*z

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])


def richardson_analytical(mu, g_L, x_L2, Az, n=1):

    c2 = (1/g_L**3)*(mu + (1-mu)*g_L**3/(1-g_L)**3)
    c3 = (1/g_L**3)*(mu - (1-mu)*g_L**4/(1-g_L)**4)
    c4 = (1/g_L**3)*(mu + (1-mu)*g_L**5/(1-g_L)**5)
    lam = fsolve(func=lambda x: x**4 + (c2 - 2)*x**2 - (c2 - 1)*(1 + 2*c2), x0=2.0)[0]
    k = (2*lam)/(lam**2 + 1 - c2)
    d1 = ((3*lam**2)/k)*(k*(6*lam**2 - 1) - 2*lam)
    d2 = ((8*lam**2)/k)*(k*(11*lam**2 - 1) - 2*lam)
    a21 = (3*c3*(k**2 - 2))/(4*(1+2*c2))
    a22 = 3*c3/(4*(1+2*c2))
    a23 = (-3*c3*lam/(4*k*d1))*((3*k**3)*lam - 6*k*(k - lam) + 4)
    a24 = (-3*c3*lam/(4*k*d1))*(2+3*k*lam)
    d21 = -c3 / (2*lam**2)
    d31 = (3/(64*lam**2))*(4*c3*a24 + c4)
    d32 = (3/(64*lam**2))*(4*c3*(a23 - d21) + c4*(4 + k**2))
    b21 = (-3*c3*lam/(2*d1))*(3*k*lam - 4)
    b22 = 3*c3*lam/d1
    b31 = (3/(8*d2))*(8*lam*(3*c3*(k*b21 - 2*a23) - c4*(2 + 3*k**2)) \
                      + (9*lam**2 + 1 + 2*c2)*(4*c3*(k*a23 - b21) + k*c4*(4 + k**2)))
    b32 = (1/d2)*(9*lam*(c3*(k*b22 + d21 - 2*a24) - c4) \
                  + (3/8)*(9*lam**2 + 1 + 2*c2)*(4*c3*(k*a24 - b22) + k*c4))
    a31 = (-9*lam/(4*d2))*(4*c3*(k*a23 - b21) + k*c4*(4 + k**2)) \
           + ((9*lam**2 + 1 - c2)/(2*d2))*(3*c3*(2*a23 - k*b21) + c4*(2+3*k**2))
    a32 = (-1/d2)*((9*lam/4)*(4*c3*(k*a24 - b22) + k*c4) + \
                   (3/2)*(9*lam**2 + 1 - c2)*(c3*(k*b22 + d21 - 2*a24) - c4))
    s1 = (1/(2*lam*(lam*(1 + k**2) - 2*k))) \
        * ((3/2)*c3*(2*a21*(k**2 - 2) - a23*(k**2 + 2) - 2*k*b21) \
           - (3/8)*c4*(3*k**4 - 8*k**2 + 8))
    s2 = (1/(2*lam*(lam*(1 + k**2) - 2*k))) \
        * ((3/2)*c3*(2*a22*(k**2 - 2) + a24*(k**2 + 2) + 2*k*b22 + 5*d21) \
           + (3/8)*c4*(12 - k**2))
   
    a1 = (-3/2)*c3*(2*a21 + a23 + 5*d21) - (3/8)*c4*(12 - k**2)
    a2 = (3/2)*c3*(a24 - 2*a22) + (9/8)*c4
    l1 = a1 + (2*lam**2)*s1
    l2 = a2 + (2*lam**2)*s2
    delta = lam**2 - c2
    Ax = ((-l2*Az**2 - delta)/l1)**(1/2)
    Ay = k*Ax
    omega = 1 + s1*Ax**2 + s2*Az**2
    theta = 0

    x0 = a21*Ax**2 + a22*Az**2 - Ax*np.cos(theta) + (a23*Ax**2 - a24*Ax**2)*np.cos(2*theta) \
        + (a31*Ax**3 - a32*Ax*Az**2*np.cos(3*theta))
    z0 = Ax*np.cos(theta) + d21*Ax*Az*(np.cos(2*theta) - 3) + (d32*Az*Ax**2 -d31*Az**3)*np.cos(3*theta)
    ydot0 = k*Ax*(lam*omega*n*np.cos(theta)) + (b21*Ax**2 - b22*Az**2) \
            * (2*lam*omega*n*np.cos(2*theta)) + (b31*Ax**3 - b32*Ax*Az**2) \
            * (3*lam*omega*n*np.cos(3*theta))

    x0 = x0 * g_L + x_L2
    z0 = z0 * g_L
    ydot0 = ydot0 * g_L

    state0 = np.array([x0, 0, z0, 0, ydot0, 0])
    T = 2*np.pi / omega

    return state0, T


M_E = 5.972e24
M_S = 1.989e30
r12 = AU = 149_597_870_700
mu = M_E / (M_E + M_S)

u = solve_l2point(mu)
x_l2 = u + 1 - mu
g_L = u

Az_km = 20_000
Az = Az_km / r12

state0, T = richardson_analytical(mu, g_L, x_l2, Az, n=1)
print(state0)
print(T)

n_half = 400
sol = solve_ivp(
        f,
        args=(mu,),
        t_span = (0.0, 1*T),
        y0 = state0,
        method = 'DOP853',
        t_eval=1.0 * T * np.linspace(0.0, 1.0, 2*n_half+1),
        rtol = 1e-11,
        atol = 1e-12,
        dense_output = False)

if 1 == 1:
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    x,  y,  z  = sol.y[:3]
    im = ax.scatter(x, y, z, c=sol.t)

    ax.plot([0.0-mu], [0], [0], 'ob', markersize=20)
    ax.plot([1.0-mu], [0], [0], 'og', markersize=12)
    ax.plot([x_l2], [0], [0], 'ok', markersize=8)

    #ax.set_xlim(0.7, 1.25)
    #ax.set_ylim(-0.225, 0.225)
    #ax.set_zlim(-0.15, 0.40)
    ax.text(x_l2, 0, -0.05, "L2", fontsize=14, horizontalalignment='center')
    #fig.colorbar(im, ax)
    nplot    = 80
    thetas   = np.linspace(0, 2*np.pi, nplot+1)[:-1]
    azimuths = -90 + 10.0 * np.cos(thetas)
    plt.show()
