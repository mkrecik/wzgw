from time_transformations import julday, GMST
from funkcje1 import *
import numpy as np

import matplotlib.pyplot as plt
import sys
import math
sys.path.append('C:\\Users\\majak\\Desktop\\sem3\\wzgw\\cw1')

# Data
nr_gwiazdy = 899

# Horizontal Coordinates
# rektascenzja
alpha_hms = [23, 55, 34.219]
alpha_h = dms2deg(alpha_hms)
alph = hms2rad(alpha_hms)
# deklinacja
delta_hms = [57, 37, 48.71]
delta = dms2rad(delta_hms)

# Sun
alpha_s = dms2deg([6, 37, 43.973])
delta_s = dms2rad([23, 8, 11.85])

# Moon
alpha_m = dms2deg([16, 9, 45.978])
delta_m = dms2rad([-23, 54, 48.69])

# Warsaw
phi_war = 52.0
phi_w = np.deg2rad(phi_war)
lambd_war = 21.0
lambd_w = np.deg2rad(lambd_war)

# Equatorial Coordinates
phi_r = np.deg2rad(0.0)
lambd_r = np.deg2rad(21.0)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(projection = '3d')

# promień Ziemi
r = 1
# siatka wspołrzędnych
u, v = np.mgrid[0:(2 * np.pi+0.1):0.1, 0:np.pi:0.1]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
z[z<0] = 0		# bez tego, narysowalibyśmy całą kulę, a chcemy tylko półkulę
ax.plot_surface(x,y,z, alpha = 0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# Set NESW labels
ax.text(0, 0, 1.1, 'N', ha='center', va='center')
ax.text(1.1, 0, 0, 'E', ha='center', va='center')
ax.text(-1.1, 0, 0, 'W', ha='center', va='center')

# Skyplot
skyplot = plt.figure(figsize = (8,8))
ax_sky = skyplot.add_subplot(polar = True)
ax_sky.set_theta_zero_location('N') # ustawienie kierunku północy na górze wykresu
ax_sky.set_theta_direction(-1)

ax_sky.set_yticks(range(0, 90+10, 10))                   # Define the yticks

yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
ax_sky.set_yticklabels(yLabel)
ax_sky.set_rlim(0, 90)

jd = julday(2023, 7, 1, 0)

azimuths = []
altitudes = []
hours = []

azimuths_sun = []
altitudes_sun = []

azimuths_moon = []
altitudes_moon = []


# Calculate LST for 24 h in Warsaw
for i in range(24):
    lst = GMST(jd) + lambd_w + np.deg2rad(15*i) + 2 

    lst_ = GMST(jd + i / 24) + lambd_w / 15
    t = (lst_ - alpha_h) * 15
    t = math.radians(t)
    t_s = (lst_ - alpha_s) * 15
    t_s = math.radians(t_s)
    t_m = (lst_ - alpha_m) * 15
    t_m = math.radians(t_m)

    # Calculate Hourly Coordinates
    '''
    Az = np.arctan2(-np.cos(delta) * np.sin(t), np.cos(phi_w) * np.sin(delta) - np.sin(phi_w) * np.cos(delta) * np.cos(t))
    h = np.arcsin(np.sin(phi_w) * np.sin(delta) + np.cos(phi_w) * np.cos(delta) * np.cos(t))
    z = np.pi/2 - h

    Az_sun = np.arctan2(-np.cos(delta_s) * np.sin(t_s), np.cos(phi_w) * np.sin(delta_s) - np.sin(phi_w) * np.cos(delta_s) * np.cos(t_s))
    h_sun = np.arcsin(np.sin(phi_w) * np.sin(delta_s) + np.cos(phi_w) * np.cos(delta_s) * np.cos(t_s))
    
    Az_moon = np.arctan2(-np.cos(delta_m) * np.sin(t_m), np.cos(phi_w) * np.sin(delta_m) - np.sin(phi_w) * np.cos(delta_m) * np.cos(t_m))
    h_moon = np.arcsin(np.sin(phi_w) * np.sin(delta_m) + np.cos(phi_w) * np.cos(delta_m) * np.cos(t_m))
    
    '''

    Az = np.arctan2(-np.cos(delta) * np.sin(t), np.cos(phi_r) * np.sin(delta) - np.sin(phi_r) * np.cos(delta) * np.cos(t))
    h = np.arcsin(np.sin(phi_r) * np.sin(delta) + np.cos(phi_r) * np.cos(delta) * np.cos(t))
    z = np.pi/2 - h

    Az_sun = np.arctan2(-np.cos(delta_s) * np.sin(t_s), np.cos(phi_r) * np.sin(delta_s) - np.sin(phi_r) * np.cos(delta_s) * np.cos(t_s))
    h_sun = np.arcsin(np.sin(phi_r) * np.sin(delta_s) + np.cos(phi_r) * np.cos(delta_s) * np.cos(t_s))
    
    Az_moon = np.arctan2(-np.cos(delta_m) * np.sin(t_m), np.cos(phi_r) * np.sin(delta_m) - np.sin(phi_r) * np.cos(delta_m) * np.cos(t_m))
    h_moon = np.arcsin(np.sin(phi_r) * np.sin(delta_m) + np.cos(phi_r) * np.cos(delta_m) * np.cos(t_m))

    # Calculate Hourly Coordinates in degrees
    Az_deg = np.rad2deg(Az)
    h_deg = np.rad2deg(h)
    z_deg = np.rad2deg(z)

    azimuths.append(Az_deg)
    altitudes.append(h_deg)
    hours.append(i)

    azimuths_sun.append(np.rad2deg(Az_sun))
    altitudes_sun.append(90 - np.rad2deg(h_sun))

    azimuths_moon.append(np.rad2deg(Az_moon))
    altitudes_moon.append(90 - np.rad2deg(h_moon))

    # narysowanie punktu na sferze
    gx = r * np.sin(Az) * np.cos(h)
    gy = r * np.cos(Az) * np.cos(h)
    gz = r * np.sin(h)
    ax.plot3D(gx,gy,gz, 'o', markersize = 5, color = 'blue')
    #ax.scatter(gx,gy,gz)
    ax_sky.scatter(Az, 90-np.rad2deg(h), color = 'blue')

    gx = r * np.sin(Az_sun) * np.cos(h_sun)
    gy = r * np.cos(Az_sun) * np.cos(h_sun)
    gz = r * np.sin(h_sun)
    ax.plot3D(gx,gy,gz, 'o', markersize = 5, color = 'orange')
    #ax.scatter(gx,gy,gz)
    ax_sky.scatter(Az_sun, 90-np.rad2deg(h_sun), color = 'orange')

    gx = r * np.sin(Az_moon) * np.cos(h_moon)
    gy = r * np.cos(Az_moon) * np.cos(h_moon)
    gz = r * np.sin(h_moon)
    ax.plot3D(gx,gy,gz, 'o', markersize = 5, color = 'grey')
    #ax.scatter(gx,gy,gz)
    ax_sky.scatter(Az_moon, 90-np.rad2deg(h_moon), color = 'grey')


ax.legend(['Sfera niebieska','Gwiazda 899', 'Słońce', 'Księżyc'])
ax_sky.legend(['Gwiazda 899', 'Słońce', 'Księżyc'])

# Wykres liniowy zależności azymutu i wysokości od czasu
plt.figure(figsize=(10, 5))
plt.plot(hours, azimuths, label='Wysokość', color = 'navy')
plt.plot(range(24), azimuths_sun, label='Słońce', color = 'red')
plt.plot(range(24), azimuths_moon, label='Księżyc', color = 'grey')
plt.xticks(range(24))
plt.xlabel('Godzina')
plt.ylabel('Stopnie')
plt.legend()
plt.title('Azymut gwiazdy na równiku w ciągu doby')

plt.figure(figsize=(10, 5))
plt.plot(range(24), altitudes, label='Gwiazda 899', color = 'navy')
plt.plot(range(24), altitudes_sun, label='Słońce', color = 'red')
plt.plot(range(24), altitudes_moon, label='Księżyc', color = 'grey')
plt.xticks(range(24))
plt.yticks(np.arange(-180, 180, 30))
plt.xlabel('Godzina')
plt.ylabel('Stopnie')
plt.legend()
plt.title('Wysokośc gwiazd na równiku w ciągu doby')

plt.show()



