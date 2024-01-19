import numpy as np
import matplotlib.pyplot as plt

# Funkcje udostępnione na zajęciach
def dms2deg(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]
    
    deg = d+m/60+s/3600
    return deg

def dms2rad(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]
    
    deg = d+m/60+s/3600
    rad = np.deg2rad(deg)
    return rad

def julday(y, m, d, h):
    if m <= 2:
        y = y - 1
        m = m + 12
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5
    return jd

def GMST(jd):
    T = (jd - 2451545) / 36525
    Tu = jd - 2451545
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2-T**3/38710000
    g = (g%360) / 15
    return g

# Funkcje własne
def horizontal_coords(alpha, delta, phi, L, gmst):
    H = (gmst * 15 + L - alpha * 15) % 360
    H = np.radians(H)
    phi = np.radians(phi)
    delta = np.radians(delta)

    h = np.arcsin(np.sin(delta)*np.sin(phi) + np.cos(delta)*np.cos(phi)*np.cos(H))
    A = np.arccos((np.sin(delta) - np.sin(phi)*np.sin(h)) / (np.cos(phi)*np.cos(h)))

    h = np.degrees(h)
    A = np.degrees(A)

    H = np.degrees(H)
    A = np.where(H > 180, 360 - A, A)

    return h, A

# Wprowadź współrzędne gwiazdy
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
    'Równika': {'phi': 0, 'L': 21},
    'Warszawy': {'phi': 52, 'L': 21},
}

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

# Tworzenie wykresów
for location, coords in locations.items():
    phi = coords['phi']
    L = coords['L']

    for star in FK5.values():
        # Obliczanie lokalnych współrzędnych horyzontalnych co godzinę
        hours = np.arange(18, 19, 1)
        h_values = []
        A_values = []

        alpha = dms2deg(star['alpha'])
        delta = dms2deg(star['delta'])
        star = list(FK5.keys())[list(FK5.values()).index(star)]
        color = colors[color_index]

        for hour in hours:
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

        ax.scatter(gx, gy, gz, label = star, color = color)
        ax.annotate(star, (gx, gz), textcoords="offset points", xytext=(15 ,5), ha='center')

        ax2.scatter(np.radians(A_values), 90 - h_values, label = star, color = color)
        ax2.annotate(star, (np.radians(A_values), 90 - h_values), textcoords="offset points", xytext=(15,5), ha='center', fontsize = 8)
        
        color_index += 1
        
    ax.set_title(f'Sfera Niebieska dla {location}')
    ax2.set_title(f'Skyplot dla {location} - godzina 18:00')
    #ax2.legend()
    #ax.legend()
    color_index = 0

    plt.show()


    '''
    # Wykres liniowy zależności wysokości od czasu
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, h_values, label = 'Wysokość', color = 'navy')
    ax.plot(hours, A_values, label = 'Azymut', color = 'lightblue')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Wysokość i azymut gwiazdy 899 w funkcji czasu dla {location}')
    ax.set_xlabel('Czas [h UTC+2]')
    ax.set_ylabel('Stopnie [°]')
    ax.legend()

    figs, axs = plt.subplots(figsize=(10, 5))
    axs.plot(hours, hs_values, label = 'Wysokość', color = 'red')
    axs.plot(hours, As_values, label = 'Azymut', color = 'orange')
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.set_title(f'Wysokość i azymut Słońca w funkcji czasu dla {location}')
    axs.set_xlabel('Czas [h UTC+2]')
    axs.set_ylabel('Stopnie [°]')
    axs.legend()

    figm, axm = plt.subplots(figsize=(10, 5))
    axm.plot(hours, hm_values, label = 'Wysokość', color = 'grey')
    axm.plot(hours, Am_values, label = 'Azymut', color = 'black')
    axm.xaxis.set_major_locator(MaxNLocator(integer=True))
    axm.set_title(f'Wysokość i azymut Księżyca w funkcji czasu dla {location}')
    axm.set_xlabel('Czas [h UTC+2]')
    axm.set_ylabel('Stopnie [°]')
    axm.legend()
    '''