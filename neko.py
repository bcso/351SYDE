from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import tan, cos, sin, pi
from scipy.integrate import odeint

##############
## y0 = yk 
## y1 = theta
## y2 = px
## y3 = py
##############

def model(y, t):
    yk, theta, vx, vy = y
    
    # constants
    m = 0.68 # Mass of pendulum
    k = 300 # Stiffness of Spring
    Y = 0
    b = 0.5 # Torsional Resistance
    R = 0.8  # Friction
    L = 0.61  # Length of Pendulum
    g = 9.81 # Gravitional acceleration

    # in between terms
    disp = (yk - Y)

    d_yk = vy + ( (L**2) * sin(theta) * cos(theta) * vx) 
    d_theta = L * cos(theta) * vx 
    d_vy = g + ( -R * d_yk - k * yk)
    d_vx = (L * cos(theta)) / ( 1 +  ((L**4)/12) * (cos(theta) ** 2)) * ((L**3 / 12 * d_theta * sin(theta) * vx) + (L * sin(theta) * d_vy) - (L * sin(theta) * g) - (d_theta * b / m))

    return [d_yk, d_theta, d_vx, d_vy]


time = np.linspace(0.0, 20.0, 100)
yinit = [0, pi/4, 0, 0]
y = odeint(model, yinit, time)

plt.figure(1)

plt.subplot(211)
plt.plot(time, y[:,0], 'b', label="yk")
plt.xlabel('t')
plt.ylabel('Motion of Spring')
plt.legend()

plt.subplot(212)
plt.plot(time, y[:,1], 'r', label="theta")
plt.xlabel('t')
plt.ylabel('Angle of rotation')
plt.legend()
plt.show()

plt.figure(2)

plt.subplot(211)
plt.plot(time, y[:,2], 'g', label="px")
plt.xlabel('t')
plt.ylabel('Displacement of Pendulum in X')
plt.legend()

plt.subplot(212)
plt.plot(time, y[:,3], 'y', label="py")
plt.xlabel('t')
plt.ylabel('Displacement of Pendulum in Y')
plt.legend()
plt.show()