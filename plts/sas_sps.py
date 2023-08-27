#/bin/env python                                                                    
'''                                                                                 
Python scrpit to generate figure for first manuscript                               
'''

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from os import path
import numpy as np
import xarray as xr

def arrange(array): # Rearranges plots from linear RRTM array scheme to linear array
        array1 = array[1:14]
        array2 = np.zeros(len(array))
        array1 = np.flip(array1)
        for i in range(len(array1)):
                array2[i] = array1[i]
        array2[13] = array[14]
        array2[14] = array[0]
        return array2

def arr_alb(array): # Rearranges albedo from linear RRTM array scheme to linear array
        bnds = np.linspace(15, 29, 15)
        diff = array
        diff = np.array(diff)
        diff = np.flip(diff)
        a = diff[0]
        for i in range(len(bnds)- 2):
                diff[i] = diff[i+1]
        diff[13] = a
        diff_up = diff
        return diff


########## VARY SOLAR ZENITH ANGLE #############
toa  = np.array(( 6,  64, 122, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869)) 

path = '/Users/jtolento/ToZ23_ppr/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_sza/'
filename_spc = np.array(('out_sas_spc89', 'out_sas_spc8', 'out_sas_spc7', 'out_sas_spc6', 'out_sas_spc5', 'out_sas_spc4', 'out_sas_spc3', 'out_sas_spc2', 'out_sas_spc1', 'out_sas_spc0'))

filename_brd = np.array(('out_sas_brd89', 'out_sas_brd8', 'out_sas_brd7', 'out_sas_brd6', 'out_sas_brd5', 'out_sas_brd4', 'out_sas_brd3', 'out_sas_brd2', 'out_sas_brd1', 'out_sas_brd0'))

toa  = np.array(( 6,  64, 122, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869)) #SURFACE FOR SAS IN 'OUT' FILE

toa = 6
surf = 57

clr_sas_spc_up_boa = np.zeros(len(filename_spc))
clr_sas_spc_dn_boa = np.zeros(len(filename_spc))
clr_sas_spc_nt_boa = np.zeros(len(filename_spc))
clr_sas_brd_up_boa = np.zeros(len(filename_spc))
clr_sas_brd_dn_boa = np.zeros(len(filename_spc))
clr_sas_brd_nt_boa = np.zeros(len(filename_spc))
clr_sas_spc_up_toa = np.zeros(len(filename_spc))
clr_sas_spc_dn_toa = np.zeros(len(filename_spc))
clr_sas_spc_nt_toa = np.zeros(len(filename_spc))
clr_sas_brd_up_toa = np.zeros(len(filename_spc))
clr_sas_brd_dn_toa = np.zeros(len(filename_spc))
clr_sas_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        clr_sas_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_sas_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_sas_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_sas_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_sas_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_sas_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        clr_sas_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_sas_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_sas_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_sas_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_sas_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_sas_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((89,80,70,60,50,40,30,20,10,0))
diff_net_boa = clr_sas_spc_nt_boa - clr_sas_brd_nt_boa
diff_net_toa = clr_sas_spc_nt_toa - clr_sas_brd_nt_toa
diff_net_boa = np.flip(diff_net_boa)
diff_net_toa = np.flip(diff_net_toa)
c = np.flip(c)
diff_net_atm = diff_net_toa - diff_net_boa


fig, axs = plt.subplots(2, 2, figsize=(12,10))
axs[0,0].plot(c, diff_net_boa, marker='o', label='BOA')
#axs[0,0].plot(c, diff_net_toa, marker='o', label='TOA')
axs[0,0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[0,0].set_title('Solar Zenith Angle')
axs[0,0].set(xlabel='Solar Zenith Angle', ylabel='Change in Net flux [W/m2]')


#diff_net_boa = diff_net_boa/np.cos(np.deg2rad(c))
#diff_net_toa = diff_net_toa/np.cos(np.deg2rad(c))
#plt.plot(c, diff_net_boa, marker='o', label='BOA')
#plt.plot(c, diff_net_toa, marker='o', label='TOA')
#plt.xlabel('Solar Zenith Angle')
#plt.ylabel('Change in Net flux [W/m2]')
#plt.title('Cosine Weighted Change in Net Flux vs Solar Zenith Angle')
#plt.legend()
#plt.show()


### CLOUD OPTICAL DEPTH ###

path = '/Users/jtolento/ToZ23_ppr/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_tau/'
filename_brd = np.array(('out_sas_brd0', 'out_sas_brdpt1', 'out_sas_brdpt2', 'out_sas_brdpt3', 'out_sas_brdpt4', 'out_sas_brdpt5', 'out_sas_brdpt7', 'out_sas_brd1', 'out_sas_brd1pt5', 'out_sas_brd2', 'out_sas_brd2pt5', 'out_sas_brd3','out_sas_brd4', 'out_sas_brd6', 'out_sas_brd8', 'out_sas_brd10', 'out_sas_brd15','out_sas_brd25','out_sas_brd35','out_sas_brd55','out_sas_brd75','out_sas_brd100'))
filename_spc = np.array(('out_sas_spc0', 'out_sas_spcpt1', 'out_sas_spcpt2', 'out_sas_spcpt3', 'out_sas_spcpt4', 'out_sas_spcpt5', 'out_sas_spcpt7', 'out_sas_spc1', 'out_sas_spc1pt5', 'out_sas_spc2', 'out_sas_spc2pt5', 'out_sas_spc3','out_sas_spc4', 'out_sas_spc6', 'out_sas_spc8', 'out_sas_spc10', 'out_sas_spc15','out_sas_spc25','out_sas_spc35','out_sas_spc55','out_sas_spc75','out_sas_spc100'))

cld_sas_spc_up_boa = np.zeros(len(filename_spc))
cld_sas_spc_dn_boa = np.zeros(len(filename_spc))
cld_sas_spc_nt_boa = np.zeros(len(filename_spc))
cld_sas_brd_up_boa = np.zeros(len(filename_spc))
cld_sas_brd_dn_boa = np.zeros(len(filename_spc))
cld_sas_brd_nt_boa = np.zeros(len(filename_spc))

cld_sas_spc_up_toa = np.zeros(len(filename_spc))
cld_sas_spc_dn_toa = np.zeros(len(filename_spc))
cld_sas_spc_nt_toa = np.zeros(len(filename_spc))
cld_sas_brd_up_toa = np.zeros(len(filename_spc))
cld_sas_brd_dn_toa = np.zeros(len(filename_spc))
cld_sas_brd_nt_toa = np.zeros(len(filename_spc))

for j in range( len(filename_spc) ) :
        cld_sas_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        cld_sas_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0,15.0,25.0,35.0,55.0,75.0,100))
diff_net_boa = cld_sas_spc_nt_boa - cld_sas_brd_nt_boa
diff_net_toa = cld_sas_spc_nt_toa - cld_sas_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa
axs[0,1].set_xscale('log')
#axs[0,1].set_xscale('symlog')
axs[0,1].plot(c[1:], diff_net_boa[1:], marker='o', label='BOA')
axs[0,1].plot(c[1:], diff_net_atm[1:], marker='o', label='ATM')
axs[0,1].set_title('Cloud Optical Depth')
axs[0,1].set(xlabel='Cloud Optical Depth', ylabel='Change in Net flux [W/m2]')


### Ice Grain  Radius  ###

path = '/Users/jtolento/ToZ23_ppr/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rds/'
filename_brd = np.array(('out_sas_brd1','out_sas_brd2','out_sas_brd3','out_sas_brd4','out_sas_brd5','out_sas_brd6','out_sas_brd7','out_sas_brd8','out_sas_brd9','out_sas_brd10'))
filename_spc = np.array(('out_sas_spc1','out_sas_spc2','out_sas_spc3','out_sas_spc4','out_sas_spc5','out_sas_spc6','out_sas_spc7','out_sas_spc8','out_sas_spc9','out_sas_spc10'))

cld_sas_spc_up_boa = np.zeros(len(filename_spc))
cld_sas_spc_dn_boa = np.zeros(len(filename_spc))
cld_sas_spc_nt_boa = np.zeros(len(filename_spc))
cld_sas_brd_up_boa = np.zeros(len(filename_spc))
cld_sas_brd_dn_boa = np.zeros(len(filename_spc))
cld_sas_brd_nt_boa = np.zeros(len(filename_spc))

cld_sas_spc_up_toa = np.zeros(len(filename_spc))
cld_sas_spc_dn_toa = np.zeros(len(filename_spc))
cld_sas_spc_nt_toa = np.zeros(len(filename_spc))
cld_sas_brd_up_toa = np.zeros(len(filename_spc))
cld_sas_brd_dn_toa = np.zeros(len(filename_spc))
cld_sas_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        cld_sas_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        
        cld_sas_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((100,200,300,400,500,600,700,800,900,1000))
diff_net_boa = cld_sas_spc_nt_boa - cld_sas_brd_nt_boa
diff_net_toa = cld_sas_spc_nt_toa - cld_sas_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa
#print(diff_net_atm)
axs[1,0].plot(c, diff_net_boa, marker='o', label='BOA')
axs[1,0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[1,0].set_title('Air Bubble Radius')
axs[1,0].set(xlabel='Air Bubble Radius [$\mu m$]', ylabel='Change in Net flux [W/m2]')

### Water Vapor Concetration ###
path = '/Users/jtolento/ToZ23_ppr/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_wvc/'
filename_brd = np.array(('out_sas_brd1','out_sas_brd2','out_sas_brd3','out_sas_brd4','out_sas_brd5','out_sas_brd6','out_sas_brd7','out_sas_brd8','out_sas_brd9','out_sas_brd100'))
filename_spc = np.array(('out_sas_spc1','out_sas_spc2','out_sas_spc3','out_sas_spc4','out_sas_spc5','out_sas_spc6','out_sas_spc7','out_sas_spc8','out_sas_spc9','out_sas_spc100'))
c = np.array((10,20,30,40,50,60,70,80,90,100))

cld_sas_spc_up_boa = np.zeros(len(filename_spc))
cld_sas_spc_dn_boa = np.zeros(len(filename_spc))
cld_sas_spc_nt_boa = np.zeros(len(filename_spc))
cld_sas_brd_up_boa = np.zeros(len(filename_spc))
cld_sas_brd_dn_boa = np.zeros(len(filename_spc))
cld_sas_brd_nt_boa = np.zeros(len(filename_spc))

cld_sas_spc_up_toa = np.zeros(len(filename_spc))
cld_sas_spc_dn_toa = np.zeros(len(filename_spc))
cld_sas_spc_nt_toa = np.zeros(len(filename_spc))
cld_sas_brd_up_toa = np.zeros(len(filename_spc))
cld_sas_brd_dn_toa = np.zeros(len(filename_spc))
cld_sas_brd_nt_toa = np.zeros(len(filename_spc))

for j in range( len(filename_spc) ) :
        cld_sas_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        cld_sas_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_sas_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_sas_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_sas_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

diff_net_boa = cld_sas_spc_nt_boa - cld_sas_brd_nt_boa
diff_net_toa = cld_sas_spc_nt_toa - cld_sas_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa
spc_atm_abs = cld_sas_spc_nt_toa - cld_sas_spc_nt_boa
spc_boa_abs = cld_sas_spc_dn_boa - cld_sas_spc_up_boa
spc_boa_alb = cld_sas_spc_up_boa / cld_sas_spc_dn_boa
print(diff_net_boa)
print((diff_net_boa / cld_sas_brd_nt_boa ) *100)
print(diff_net_atm)
axs[1,1].plot(c, diff_net_boa, marker='o', label='BOA')
axs[1,1].plot(c, diff_net_atm, marker='o', label='ATM')
axs[1,1].set_title('Water Vapor Concentration')
axs[1,1].set(xlabel='Percent Humidity [%]', ylabel='Change in Net flux [W/m2]')
fig.suptitle('Ice - Single Parameter Sensitivity', fontsize='16',fontweight='bold')
plt.legend()                


plt.savefig('/Users/jtolento/Desktop/ppr1/sas_sps.eps')
plt.show()

