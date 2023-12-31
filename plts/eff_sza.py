#!/usr/bin/env python
"""
Calculate effective solar zenith angle for TZW23
"""

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import scipy.integrate as integrate
import scipy.special as special

data1_path = "/Users/jtolento/TZW23/plts/"
fl         = "merra2_ANN_200001_200912_climo.nc"
data1 = Dataset(data1_path+fl)


Fs_toa = 1361           #toa solar flux [W/m^2]
SWTDN  = data1['SWTDN'] #toa_incoming_shortwave_flux
lat    = data1['lat']
lon    = data1['lon']
nbr_lat  = len(lat)
nbr_lon  = len(lon) 
sw_toa = np.zeros((nbr_lat, nbr_lon))

# Calculate zonal average
for i in range(nbr_lat):
    for j in range(nbr_lon):
        sw_toa[i,j] = SWTDN[0, i, j]

znl_avg = np.zeros(nbr_lat)
for i in range(nbr_lat):
    for j in range(nbr_lon):
        znl_avg[i] = znl_avg[i] +  sw_toa[i,j]
    znl_avg[i] = znl_avg[i] / nbr_lon

#znl_avg = np.genfromtxt('znl_avg.txt')
znl_avg = 2*znl_avg
#plt.plot(lat[:], znl_avg)
#plt.show()

znl_avg30 = 0
c         = 0


for i in range(nbr_lat):
    if (lat[i]>0 and lat[i]<30):
        znl_avg30 = znl_avg30 + (znl_avg[i]*np.cos(np.radians(lat[i])))
        c = c+np.cos(np.radians(lat[i]))
znl_avg30 = znl_avg30 / c
eff_30 = np.arccos(znl_avg30 / Fs_toa)
eff_30 = np.rad2deg(eff_30)
print(eff_30)        


znl_avg60 = 0
c         = 0
for i in range(nbr_lat):
    if (lat[i]>30 and lat[i]<60):
        znl_avg60 = znl_avg60 + znl_avg[i]*np.cos(np.radians(lat[i]))
        c = c+np.cos(np.radians(lat[i]))


#print(znl_avg60)
#print(c)


znl_avg60 = znl_avg60 / c
#print(znl_avg60)
eff_60 = np.arccos(znl_avg60 / Fs_toa)
eff_60 = np.rad2deg(eff_60)
print(eff_60)
#print(Fs_toa*np.cos(np.radians(eff_60)))

znl_avg90 = 0
c         = 0
for i in range(nbr_lat):
    if (lat[i]>60 and lat[i]<90):
        znl_avg90 = znl_avg90 + znl_avg[i]*np.cos(np.radians(lat[i]))
        c = c+np.cos(np.radians(lat[i]))

znl_avg90 = znl_avg90 / c
eff_90 = np.arccos(znl_avg90 / Fs_toa)
eff_90 = np.rad2deg(eff_90)
print(eff_90)

