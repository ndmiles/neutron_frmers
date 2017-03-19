#! /usr/bin/env python3

# A Python 3.6 port of iontraj_rk4.pro

import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit


# Define constants
PI = np.pi
qe = 1.602176462E-19              # charge of an electron (C)
epsilon_0 = 8.85e-12              # permittivity of free space
k = 1.0 / (4.0 * PI * epsilon_0)  # Coulomb force constant
me = 9.10938188E-31               # electron rest mass (kg)
mp = 1.67262158E-27               # proton rest mass (kg)

q = 200.0e3 * (2.0e-2**2.0) / k

alpha = -(k * 2.0 * qe * q) / (2.0 * mp)  # Constant used in stepping

deltat = 1.0e-11

Nelectrons = 100

n = 100000  #

x = np.zeros([n, Nelectrons], dtype=np.float64)
y = np.zeros([n, Nelectrons], dtype=np.float64)

vx = np.zeros([n, Nelectrons], dtype=np.float64)
vy = np.zeros([n, Nelectrons], dtype=np.float64)

y[0, :Nelectrons] = 0.02  # Replaces the for loop in the IDL code

seed = 1

# BELOW CODE WAS IN IDL CODE BUT COMMENTED OUT...?
# fix((systime(1))mod32000); random seed

p = np.random.uniform(0, seed, Nelectrons)
p = (p * np.array([p < 1.], dtype=np.float64)) + ((1. - 1.E-7) * np.array([p == 1.]))

E0 = 100. - (7000. * np.log(1. - p))
E0 = E0.flatten()

for i in range(Nelectrons):
    vx[0, i] = ((2.0 * E0[i] / mp)**.5) * 1.5e-10  # units of m/s
    # vy[0, i] = ((2.0 * E0[i] / mp)**.5) * 5.0e-2  # units of m/s
    # vy[0, i] = ((2.0 * E0[i] * 1.0E-30 / mp)**.5) * .5  # units of m/s


@jit(cache=True)
def rk4_method(Nelectrons, n, deltat, x, y, vx, vy, alpha):
    for j in range(n-1):
        for k in range(Nelectrons):

            k1x = vx[j, k] * deltat
            k2x = (vx[j, k] + (k1x / 2)) * deltat
            k3x = (vx[j, k] + (k2x / 2)) * deltat
            k4x = (vx[j, k] + k3x) * deltat

            x[j+1, k] = x[j, k] + (k1x + 2*k2x + 2*k3x + k4x) / 6

            k1y = vy[j, k] * deltat
            k2y = (vy[j, k] + k1y / 2) * deltat
            k3y = (vy[j, k] + k2y / 2) * deltat
            k4y = (vy[j, k] + k3y) * deltat

            y[j+1, k] = y[j, k] + (k1y + 2*k2y + 2*k3y + k4y) / 6

            temp_dvx = ((alpha * x[j, k]) / (x[j, k]**2 + y[j, k]**2)**(3/2))

            k1vx = temp_dvx * deltat
            k2vx = (temp_dvx + k1x / 2) * deltat
            k3vx = (temp_dvx + k2x / 2) * deltat
            k4vx = (temp_dvx + k3x) * deltat

            vx[j+1, k] = vx[j, k] + (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6

            temp_dvy = ((alpha * y[j, k]) / (x[j, k]**2 + y[j, k]**2)**(3/2))

            k1vy = temp_dvy * deltat
            k2vy = (temp_dvy + k1x / 2) * deltat
            k3vy = (temp_dvy + k2x / 2) * deltat
            k4vy = (temp_dvy + k3x) * deltat

            vy[j+1, k] = vy[j, k] + (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6

start = time.time()
rk4_method(Nelectrons, n, deltat, x, y, vx, vy, alpha)
print("done in %.4f seconds" % (time.time() - start))


# Ion trajectory plot
plt.figure()
# plt.plot(x, y, linewidth=2)
plt.title('Ion Trajectory inside a Coulomb Potential')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
# plt.ylim([.0199906, .0200001])
ion0 = plt.plot(x[:, 0], y[:, 0], color="purple",  linewidth=2, label='ion0')
ion1 = plt.plot(x[:, 1], y[:, 1], color="red",  linewidth=2, label='ion1')
ion2 = plt.plot(x[:, 2], y[:, 2], color="blue", linewidth=2, label='ion2')
ion3 = plt.plot(x[:, 3], y[:, 3], color="violet", linewidth=2, label='ion3')
ion4 = plt.plot(x[:, 4], y[:, 4], color="green", linewidth=2, label='ion4')
ion5 = plt.plot(x[:, 5], y[:, 5], color="orange", linewidth=2, label='ion5')
plt.savefig("ion_trajectories.png", clobber=True)
plt.legend()
plt.show()
plt.close()


# Ion velocity distribution plot
plt.figure()
plt.plot(vx[0, :], color='navy', linewidth=2)
plt.title('Initial Ion Velocity Distribution')
plt.ylabel('Velocity (m/s)')
plt.savefig('initial_ion_velocities.png')
plt.show()
plt.close()

# Store the ion_data into an array that will be written to a file
ion_data = np.zeros([len(x[:, 0]), 24])
ion_data[:, 0] = x[:, 0]
ion_data[:, 1] = y[:, 0]
ion_data[:, 2] = vx[:, 0]
ion_data[:, 3] = vy[:, 0]
ion_data[:, 4] = x[:, 1]
ion_data[:, 5] = y[:, 1]
ion_data[:, 6] = vx[:, 1]
ion_data[:, 7] = vy[:, 1]
ion_data[:, 8] = x[:, 2]
ion_data[:, 9] = y[:, 2]
ion_data[:, 10] = vx[:, 2]
ion_data[:, 11] = vy[:, 2]
ion_data[:, 12] = x[:, 3]
ion_data[:, 13] = y[:, 3]
ion_data[:, 14] = vx[:, 3]
ion_data[:, 15] = vy[:, 3]
ion_data[:, 16] = x[:, 4]
ion_data[:, 17] = y[:, 4]
ion_data[:, 18] = vx[:, 4]
ion_data[:, 19] = vy[:, 4]
ion_data[:, 20] = x[:, 5]
ion_data[:, 21] = y[:, 5]
ion_data[:, 22] = vx[:, 5]
ion_data[:, 23] = vy[:, 5]
output_file = 'ion_trajectory_data.txt'  # Name of ion trajectory data file
np.savetxt(X=ion_data, fname=output_file)  # Save the ion trajectory data

# Save the initial parameters used to produce the trajectories
ion_initial_data = np.zeros([len(x[0, :]), 2])
ion_initial_data[:, 0] = np.arange(1, 101)
ion_initial_data[:, 1] = E0
output_file = 'ion_trajectory_energy_data.txt'
np.savetxt(X=ion_initial_data, fname=output_file)

# Array of distances of the electrons (replaces FOR loop in IDL code)
re = (x[n-1, :]**2 + y[n-1, :]**2)**0.5
rmax = re.max()
deltar = .10
counter_max = int(rmax / deltar)
rho = np.zeros(counter_max)
r = np.arange(counter_max, dtype=np.float) * deltar

for counter in range(counter_max - 1):
    for k in range(Nelectrons):
        if (re[k] < r[counter+1]) & (re[k] > r[counter]):
            rho[counter] += 1

nbins = 50  # Number of bins
rr = np.zeros(nbins)
r_bins = np.arange(nbins, dtype=np.float) * 0.01
ss = np.zeros(len(r_bins))
s = PI * (r_bins**2)

for k in range(1, nbins):
    g = np.where((re >= r_bins[k-1]) & (re <= r_bins[k]))
    ng = len(g[0])
    ss = s[k] - s[k - 1]
    rr[k - 1] = ng / ss

# Create number density plot
plt.figure()
plt.plot(r_bins, rr, color='magenta', linewidth=3)
plt.title('Ion Number Density')
plt.xlabel('Radius (m)')
plt.ylabel('Number Density (m$^{-2}$)')
plt.savefig('number_density.png')
plt.show()
plt.close()

# Bin Energy Values
l = 100.0
u = 200.00
n = 15

hist = np.zeros(n)
histerr = np.zeros(n)

X = np.arange(n, dtype=float)
for k in range(n):
    g = np.where((E0 >= l) & (E0 < u))
    ng = len(g[0])
    if ng > 0:

        hist[k] = ng / (u - l)
        histerr[k] = np.sqrt(ng) / (u - l)
        X[k] = (u + l) / 2
    l = u
    u *= 2

# Plot energy density
plt.figure()
plt.plot(X, hist)
density = plt.plot(rho)
# radius = plt.plot(r)
plt.savefig('energy_density.png')
plt.close()
