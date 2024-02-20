import numpy as np
from pyproj import Proj, transform, CRS, Transformer, Geod
import plotly.graph_objects as go
import pandas as pd

# algortyrm własny -> PL2000, PL92

# Parametry elipsoidy
a = 6378137
a2 = a ** 2
e2 = 0.00669438002290

b2 = a2 * (1 - e2)
e22 = (a2- b2)  / b2
m2000 = 0.999923
m92 = 0.9993

# 
A0 = 1 - (e2 / 4) - (3 * e2 ** 2 / 64) - (5 * e2 ** 3 / 256)
A2 = (3 / 8) * (e2 + e2 ** 2 / 4 + 15 * e2 ** 3 / 128)
A4 = (15 / 256) * (e2 ** 2 + 3 * e2 ** 3 / 4)
A6 = 35 * e2 ** 3 / 3072

# Współrzędne punktów
phis = [53.75, 54.10937712005116, 54.099667067097876, 53.74028936233757] 
lambdas = [15.25, 15.25, 16.778726126645378, 16.778726126645378]
lam0 = np.radians(15)
phis = np.radians(phis)
lambdas = np.radians(lambdas)


lengths_given = [400000, 100000, 400000, 100000]

# Promienie krzywizny
def M_and_N(phi):
    sin_phi = np.sin(phi)
    M = a * (1 - e2) / (1 - e2 * sin_phi**2)**(3/2)
    N = a / np.sqrt(1 - e2 * sin_phi**2)
    return M, N

# Phi, lambda, h na X, Y, Z
def orto(p, l, h):
    N = a / (np.sqrt(1 - e2 * np.sin(l) * np.sin(l)))
    X = (N + h) * np.cos(p) * np.cos(l)
    Y = (N + h) * np.cos(p) * np.sin(l)
    Z = (N * (1 - e2) + h) * np.sin(p)
    return([X, Y, Z])

# Funkcja obliczająca macierz obrotu
def macierz_obrotu(p, l):
    macierz = np.array([[-np.sin(p) * np.cos(l), -np.sin(l), np.cos(p) * np.cos(l)],
                        [-np.sin(p) * np.sin(l), np.cos(l), np.cos(p) * np.sin(l)],
                        [np.cos(p), 0, np.sin(p)]])
    return macierz


# Phi, lambda na X, Y na płaszczyźnie Gaussa-Krügera
def pl_to_gk(phi, lamb):
    d_lamb = lamb - lam0
    t = np.tan(phi)
    eta2 = e22 * np.cos(phi) ** 2
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)

    sigma = a * (A0 * phi - A2 * np.sin(2 * phi) + A4 * np.sin(4 * phi) - A6 * np.sin(6 * phi))

    x = sigma + d_lamb ** 2 / 2 * N * np.sin(phi) * np.cos(phi) * (1 + d_lamb ** 2 / 12 * (np.cos(phi) ** 2 * (5 - t ** 2 + 9 * eta2 + 4 * eta2 ** 2) + d_lamb ** 4 / 360 * np.cos(phi) ** 4 * (61 - 58 * t ** 2 + t ** 4 + 270 * eta2 - 330 * eta2 * t ** 2)))
    y = d_lamb + N + np.cos(phi) * (1 + d_lamb ** 2 / 6 * np.cos(phi) ** 2 * (1 - t ** 2 + eta2) + d_lamb ** 4 / 120 * np.cos(phi) ** 4 * (5 - 18 * t ** 2 + t ** 4 + 14 * eta2 - 58 * eta2 * t ** 2))                                                       

    x = N * np.cos(phi) * d_lamb
    y = N * np.log(np.tan(np.pi / 4 + phi / 2))
    return x, y

# Przeliczenie odwrotne
def gk_to_phi(x, y):
    while True:
        phi1 = y / a * A0
        x, y, t, N, eta, sigma = pl_to_gk(phi, 0)
        phi = phi + (y - sigma) / a * A0
        if abs(phi1 - phi1) < 10 ** (-10):
            break
    phi = phi1 - (y ** 2 * t / (2 * a * A0 * N))

x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []

x_out = [x1, x2, x3, x4]
y_out = [y1, y2, y3, y4]

lines = ['1-2', '2-3', '3-4', '4-1']

# Pętla transformacji
for i in range(4):
    x, y = pl_to_gk(phis[i], lambdas[i])
    x2000 = m2000 * x
    y2000 = m2000 * x + 5000000 + 500000
    x92 = m92 * x + 500000
    y92 = m92 * y - 5300000
    x1.append(round(x, 2))
    y1.append(round(y, 2))
    x2.append(round(x2000, 2))
    x3.append(round(x92, 2))
    y2.append(round(y2000, 2))
    y3.append(round(y92, 2))


dfGK = pd.DataFrame({'nr':lines, 'x' : x1, 'y': y1})
print(dfGK.to_latex())

df2000 = pd.DataFrame({'nr':lines, 'x' : x2, 'y': y2})
print(df2000.to_latex())

df1992 = pd.DataFrame({'nr':lines, 'x' : x3, 'y': y3})
print(df1992.to_latex())

'''
def calculate_length_reduction(coords, m, phis, a, e2, lengths):
    reduced_lengths = []
    for i in range(len(coords)):
        xA, yA = coords[i]
        xB, yB = coords[(i+1) % len(coords)]  # to get the next point and loop back to the first for the last point
        phi_mid = np.radians((phis[i] + phis[(i+1) % len(phis)]) / 2)  # średnia szerokość geodezyjna dla środkowego punktu odcinka
        M, N = M_and_N(phi_mid, a, e2)
        Rm = np.sqrt(M * N)  # średni promień krzywizny dla odcinka
        sAB = lengths[i]
        y_square = (yA**2 + yA*yB + yB**2)
        rAB = sAB * y_square / (6 * Rm**2)  # redukcja długości
        s_elip = sAB / m - rAB  # długość odcinka na elipsoidzie
        reduced_lengths.append(s_elip)
    return reduced_lengths

reduced_lengths_2000 = calculate_length_reduction(x_out[:][0], m2000, phis, a, e2, lengths_given)
print(reduced_lengths_2000)

# Tabela z współrzędnymi punktów
fig = go.Figure(go.Table(
    header=dict(values=['Projekcja', 'Kod EPSG', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']), 
    cells=dict(values=[output_names, output_codes, x1_out, y1_out, x2_out, y2_out, x3_out, y3_out, x4_out, y4_out])
))
fig.update_layout(
    title = 'Współrzędne punktów w różnych układach współrzędnych'
)
#fig.show()


# Tabela długości
fig = go.Figure(go.Table(
    header=dict(values=['nr', 'Dane długości[m]', 'PL-2000[m]', 'Gauss-Krüger[m]', 'Elipsoida[m]']), 
    cells=dict(values=[lines, lengths_given, lengths_2000, lengths_gk2000, lengths_elip2000])))
fig.update_layout(
    title = 'Długości odcinków')
#fig.show()

fig = go.Figure(go.Table(
    header=dict(values=['nr', 'Dane długości[m]', 'PL-1992[m]', 'Gauss-Krüger[m]', 'Elipsoida[m]']), 
    cells=dict(values=[lines, lengths_given, lengths_1992, lengths_gk1992, lengths_elip1992])))
fig.update_layout(
    title = 'Długości odcinków')
#fig.show()

# Tabela azymutów
fig = go.Figure(go.Table(
    header=dict(values=['Nr', 'Azymut dany', 'Azymut Kivioj', 'Azymut Vincent', 'Azymut PL-2000', 'Azymut odw. PL-2000']),
    cells=dict(values=[lines, azimuths_given_dms, azimuths_kivioj_dms, azimuths_rect, azimuths_vincent_dms, azimuths_back_vincent_dms, azimuths_2000_dms, azimuths_rev_2000_dms])
))
fig.update_layout(
    title = 'Azymuty odcinków'
)
#fig.show()

# Tabela powierzchni
fig = go.Figure(go.Table(
    header=dict(values=['Ćw. 3', 'PL-2000', 'PL-2000', 'PL-1992', 'PL-1992']),
    cells=dict(values=[areas_ex3, P2000_1, P2000_2, P1992_1, P1992_2])
))
fig.update_layout(
    title = 'Pola powierzchni [m^2]'
)
# fig.show()
'''
