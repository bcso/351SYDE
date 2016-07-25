from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import tan, cos, sin, pi
from scipy.integrate import odeint, simps, cumtrapz 

##############
## y0 = yk 
## y1 = theta
## y2 = px
## y3 = py
##############

def model(y, t):
    yk, theta, vx, vy = y
    
    # constants
    m = 0.206 # Mass of pendulum
    k = 10 # Stiffness of Spring
    b = 0.3 # Torsional Resistance
    R = 1.5 # Friction
    L = 0.61  # Length of Pendulum
    g = 9.81 # Gravitional acceleration
    Y = 0 # Equilibrium position

    # in between terms
    disp = (yk - Y)

    d_yk = vy + ((tan(theta) * vx)) 
    d_theta = vx / (L * cos(theta)) 
    d_vy = g + (( -R * d_yk - k * yk)/m)

    # the derivative causality is resolved here, so adding some in between
    # terms for easier debugging
    e_21 = tan(theta) * (d_vy - g) # comes from the left side of bg
    e_24 = d_theta * b  # torsional resistance
    e_22 = d_theta * tan(theta) * vx / (12 * (cos(theta)**2))
    factor = 1 / (1 + (1 / ( 12 * (cos(theta)**2))))

    d_vx = factor * (e_21 - e_22 - e_24)
    return [d_yk, d_theta, d_vx, d_vy]

time = np.linspace(0.0, 8.0, 10000)
# Initial condition parameters
# yinit = [Vertical spring displacement, pendulum angle relative to vertical, horizontal velocity, vertical]
yinit = [0, pi/4, 0, 0]
y = odeint(model, yinit, time)

# the state equations give us velocity
# integrate again to get displacement
# our variable of interest
ped_y = cumtrapz(y[:,3], time, initial=0)
ped_x = cumtrapz(y[:,2], time, initial=0)

plt.figure(1)

plt.subplot(311)
plt.plot(time, y[:,0])
plt.xlabel('t [s]')
plt.ylabel('Displacement [m]')
plt.title('Displacement of Spring in Y')
plt.grid()
plt.legend()

# plt.subplot(312)
# plt.plot(time, y[:,1])
# plt.xlabel('t [s]')
# plt.ylabel('Displacement [rad]')
# plt.title('Angle of rotation')
# plt.legend()

plt.subplot(312)
plt.plot(time, ped_x)
plt.xlabel('t [s]')
plt.ylabel('Displacement [m]')
plt.title('Displacement of Pendulum in X')
plt.grid()
plt.legend()

plt.subplot(313)
plt.plot(time, ped_y)
plt.xlabel('t [s]')
plt.ylabel('Displacement [m]')
plt.title('Displacement of Pendulum in Y')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()