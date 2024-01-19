#SFERA

import matplotlib.pyplot as plt
import numpy as np

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


# narysowanie punktu na sferze
gx = r * np.sin(Az) * np.cos(h)
gy = r * np.cos(Az) * np.cos(h)
gz = r * np.sin(h)
ax.plot3D(gx,gy,gz)


#SKYPLOT

fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(polar = True)
ax2.set_theta_zero_location('N') # ustawienie kierunku północy na górze wykresu
ax2.set_theta_direction(-1)

ax.set_yticks(range(0, 90+10, 10))                   # Define the yticks

yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
ax.set_yticklabels(yLabel)
ax.set_rlim(0,90)

# narysowanie punktu na wykresie 
ax.scatter(Az, 90-np.rad2deg(h))	



