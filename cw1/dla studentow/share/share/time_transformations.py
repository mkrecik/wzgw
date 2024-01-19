# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 15:16:25 2022

@author: mgrzy
"""

import numpy as np
import math
import sys
sys.path.insert(1, 'C:\\Users\\mgrzy\\OneDrive - Politechnika Warszawska\\ZGIAG\\dydaktyka\\astronomia\\python')
from funkcje import *
def julday(y,m,d,h):
    '''
    Simplified Julian Date generator, valid only between
    1 March 1900 to 28 February 2100
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
    # A = np.trunc(y/100)
    # B = 2-A+np.trunc(A/4)
    # C = np.trunc(365.25*y)
    # D = np.trunc(30.6001 * (m+1))
    # jd = B + C + D + d + 1720994.5
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5;
    return jd

def GMST(jd):
    '''
    calculation of Greenwich Mean Sidereal Time - GMST in hours
    ----------
    jd : TYPE
        julian date
    '''
    T = (jd - 2451545) / 36525
    Tu = jd - 2451545
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2-T**3/38710000
    g = (g%360) / 15
    return g




if __name__ == '__main__':
    jd = julday(2021,3,21,0)
    print(jd)
    g = GMST(jd)
    # print(deg2dms(g))