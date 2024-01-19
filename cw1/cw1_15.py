import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
    H = (gmst * 15 + L - alpha * 15) % 360 # czas 
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

def interpolate(hours, h_values, label, color):
    xnew = np.linspace(min(hours), max(hours), 300)
    spl = make_interp_spline(hours, h_values, k=3)
    ynew = spl(xnew)
    plt.plot(xnew, ynew, label=label, color=color)
    #plt.fill_between(xnew, 0, ynew, where=(ynew > 0), color=color, alpha=0.6, label = label)

# Współrzędne gwiazd
alpha_hms = [23, 55, 34.219]
delta_hms = [57, 37, 48.71]
alpha = dms2deg(alpha_hms)
delta = dms2deg(delta_hms)

# Słońce
alpha_s = dms2deg([6, 37, 43.973])
delta_s = dms2deg([23, 8, 11.85])


# Współrzędne obserwatorów
locations = {
    'Warszawy': {'phi': 52, 'L': 21},
    'Równika': {'phi': 0, 'L': 21},
}

if __name__ == '__main__':
# Tworzenie wykresów
    for location, coords in locations.items():
        phi = coords['phi']
        L = coords['L']

        # Obliczanie lokalnych współrzędnych horyzontalnych co godzinę
        hours = np.arange(0, 25, 1)
        h_values = []
        A_values = []
        hs_values = []
        As_values = []
        hm_values = []
        Am_values = []

        for hour in hours:
            jd = julday(2023, 7, 1, hour - 1)  # UTC+2 dla Polski
            gmst = GMST(jd)
            h, A = horizontal_coords(alpha, delta, phi, L, gmst)
            hs, As = horizontal_coords(alpha_s, delta_s, phi, L, gmst)
            h_values.append(h)
            A_values.append(A)
            hs_values.append(hs)
            As_values.append(As)

        h_values = np.array(h_values)
        A_values = np.array(A_values)
        hs_values = np.array(hs_values)
        As_values = np.array(As_values)

        # Sfera Niebieska
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121, projection='3d')
        u, v = np.mgrid[0:(2 * np.pi):0.01, 0:np.pi:0.01]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        z[z < 0] = 0
        ax.plot_surface(x, y, z, alpha=0.1, color='b')
        gx = np.sin(np.radians(A_values)) * np.cos(np.radians(h_values))
        gy = np.cos(np.radians(A_values)) * np.cos(np.radians(h_values))
        gz = np.sin(np.radians(h_values))
        ax.scatter(gx, gy, gz, c=hours, cmap='viridis', label = 'FK5 899')

        gxs = np.sin(np.radians(As_values)) * np.cos(np.radians(hs_values))
        gys = np.cos(np.radians(As_values)) * np.cos(np.radians(hs_values))
        gzs = np.sin(np.radians(hs_values))
        ax.scatter(gxs, gys, gzs, label = 'Słońce', color = 'red')

        ax.set_title(f'Sfera Niebieska dla {location}')

        # Skyplot
        ax = plt.subplot(122, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_yticks(range(0, 90+10, 10))
        yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
        ax.set_yticklabels(yLabel)
        ax.set_rlim(0, 90)
        ax.scatter(np.radians(A_values), 90 - h_values, c=hours, cmap='viridis', label = 'FK5 899')
        ax.scatter(np.radians(As_values), 90 - hs_values, label = 'Słońce', color = 'red')
        ax.set_title(f'Skyplot dla {location}')
        ax.legend()

        # Wykresy zależności wysokości i azymutu od czasu
        # Azymut
        plt.figure(figsize=(10, 5))
        plt.plot(hours, (A_values), label='FK5 899', color = 'navy')
        plt.plot(hours, (As_values), label='Słońce', color = 'red')
        plt.xticks(range(25))
        plt.xlabel('Godzina [h UTC+2]')
        plt.ylabel('Azymut [°]')
        plt.yticks(np.arange(0, 361, 60))
        plt.legend()
        plt.grid()
        plt.title(f'Azymut gwiazd w ciągu doby dla {location}')

        # Wysokość
        plt.figure(figsize=(10, 5))
        interpolate(hours, h_values, label='FK5 899', color = 'navy')
        interpolate(hours, hs_values, label='Słońce', color = 'red')
        plt.xticks(range(25))
        plt.yticks(np.arange(-90, 91, 30))
        plt.xlabel('Godzina [h UTC+2]')
        plt.ylabel('Wysokość [°]')
        plt.legend()
        plt.grid()
        plt.title(f'Wysokośc gwiazd w ciągu doby dla {location}')

        # Wykres wysokość z interpolacją
        plt.figure(figsize=(14, 7))
        interpolate(hours, h_values, 'FK5 899', 'blue')
        interpolate(hours, hs_values, 'Słońce', 'red')
        plt.axhline(0, color='black',linewidth=0.5)
        plt.ylim(0, 90)
        plt.yticks(np.arange(0, 91, 10))
        plt.xticks(range(25))
        plt.xlabel('Godzina [h UTC+2]')
        plt.ylabel('Wysokość nad horyzontem [°]')
        plt.title(f'Wykres wysokości z interpolacją dla {location}')
        plt.legend()

        # Panorama
        plt.figure(figsize=(10, 5))
        plt.plot(A_values, h_values, label = 'Wysokość', color = 'navy')
        plt.plot(As_values, hs_values, label = 'Słońce', color = 'red')
        plt.ylim(0, 90)
        plt.title(f'Panorama nieba dla {location}')
        plt.xlabel('Azymut [°]')
        plt.ylabel('Wysokość [°]')
        plt.legend()

        plt.show()