import numpy as np
import matplotlib.pyplot as plt
from cw1_15 import dms2deg, julday, GMST, horizontal_coords

# Wwsółrzędne gwiazd
FK5 = {
    'Merak' : {'alpha' : [11, 3, 14.669], 'delta' : [56, 15, 21.18]}, #416
    'Dubhe ' : {'alpha' : [11, 5, 9.530], 'delta' : [61, 37, 24.44]}, #417
    'Phecda ' : {'alpha' : [11, 55, 3.388], 'delta' : [53, 33, 50.55]}, #447
    'Megrez' : {'alpha' : [12, 16, 34.755], 'delta' : [56, 54, 7.8]}, #456
    'Alioth' : {'alpha' : [12, 55, 3.395], 'delta' : [55, 49, 57.62]}, #483
    'Mizar' : {'alpha' : [13, 24, 52.075], 'delta' : [54, 48, 11.5]}, #497
    'Alkaid' : {'alpha' : [13, 48, 27.861], 'delta' : [49, 11, 48.04]}, #509
}

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
color_index = 0

# Współrzędne obserwatorów
locations = {
    'Warszawy': {'phi': 52, 'L': 21},
    'Równika': {'phi': 0, 'L': 21},
}

# Tworzenie wykresów
for location, coords in locations.items():
    # Sfera Niebieska
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121, projection='3d')
    u, v = np.mgrid[0:(2 * np.pi):0.01, 0:np.pi:0.01]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    z[z < 0] = 0
    ax.plot_surface(x, y, z, alpha=0.1, color='b')

    # Skyplot
    ax2 = plt.subplot(122, polar=True)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_yticks(range(0, 90+10, 10))
    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    ax2.set_yticklabels(yLabel)
    ax2.set_rlim(0, 90)

    color_index = 0
    phi = coords['phi']
    L = coords['L']

    ax.set_title(f'Sfera Niebieska dla {location}')
    ax2.set_title(f'Skyplot dla {location}')
    
    lines_sfera = []
    lines_skyplot = []
    
    for star_name, star_coords in FK5.items():
        alpha = dms2deg(star_coords['alpha'])
        delta = dms2deg(star_coords['delta'])
        color = colors[color_index]
        
        line_sfera, = ax.plot([], [], 'o-', color=color, label=star_name)
        line_skyplot, = ax2.plot([], [], 'o-', color=color, label=star_name)
        
        lines_sfera.append(line_sfera)
        lines_skyplot.append(line_skyplot)
        
        color_index = (color_index + 1) % len(colors)
        
    ax.legend(loc = 'upper right')
    ax2.legend(loc = 'upper right')
    
    # Obliczanie lokalnych współrzędnych horyzontalnych co godzinę
    for hour in np.arange(0, 24, 1):
        h_values_all = []
        A_values_all = []
        gx_all = []
        gy_all = []
        gz_all = []
        
        for star_coords in FK5.values():
            h_values = []
            A_values = []
            alpha = dms2deg(star_coords['alpha'])
            delta = dms2deg(star_coords['delta'])
            
            jd = julday(2023, 7, 1, hour - 1)  # UTC+2 dla Polski
            gmst = GMST(jd)
            h, A = horizontal_coords(alpha, delta, phi, L, gmst)
            h_values.append(h)
            A_values.append(A)

            h_values = np.array(h_values)
            A_values = np.array(A_values)
                
            gx = np.sin(np.radians(A_values)) * np.cos(np.radians(h_values))
            gy = np.cos(np.radians(A_values)) * np.cos(np.radians(h_values))
            gz = np.sin(np.radians(h_values))
            
            gx_all.append(gx)
            gy_all.append(gy)
            gz_all.append(gz)
            h_values_all.append(h_values)
            A_values_all.append(A_values)
            
        # Zaktualizuj dane na wykresach
        for i in range(len(FK5)):
            lines_sfera[i].set_data(gx_all[i], gy_all[i])
            lines_sfera[i].set_3d_properties(gz_all[i])
            lines_skyplot[i].set_data(np.radians(A_values_all[i]), 90 - h_values_all[i])
        
        plt.draw()
        plt.pause(0.5)


plt.show()