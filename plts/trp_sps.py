#/bin/env python                                                                    
'''                                                                                 
Python scrpit to generate figure 6 for TZW23
'''

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from os import path
import numpy as np
import xarray as xr
import matplotlib.transforms as mtransforms

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
toa  = np.array((6,   73, 140, 207, 274, 341, 408, 475, 542, 609, 676, 743, 810, 877,  944)) # TOA FOR TROP ATM IN 'OUT' FILE 
surf = np.array((66, 133, 200, 267, 334, 401, 468, 535, 602, 669, 736, 803, 870, 937, 1004)) # SURFACE FOR TROP ATM IN 'OUT' FILE

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/wtr_sza/'
#filename_spc = np.array(('out_trp_spc89', 'out_trp_spc8', 'out_trp_spc7', 'out_trp_spc6', 'out_trp_spc5', 'out_trp_spc4', 'out_trp_spc3','out_trp_spc25','out_trp_spc22','out_trp_spc2','out_trp_spc18','out_trp_spc15', 'out_trp_spc1', 'out_trp_spc05','out_trp_spc0'))

filename_spc = np.array(('out_trp_spc89', 'out_trp_spc8', 'out_trp_spc7', 'out_trp_spc6', 'out_trp_spc5', 'out_trp_spc4', 'out_trp_spc3','out_trp_spc2', 'out_trp_spc1','out_trp_spc0'))

#filename_brd = np.array(('out_trp_brd89', 'out_trp_brd8', 'out_trp_brd7', 'out_trp_brd6', 'out_trp_brd5', 'out_trp_brd4', 'out_trp_brd3','out_trp_brd25','out_trp_brd22','out_trp_brd2', 'out_trp_brd18','out_trp_brd15','out_trp_brd1', 'out_trp_brd05','out_trp_brd0'))
filename_brd = np.array(('out_trp_brd89', 'out_trp_brd8', 'out_trp_brd7', 'out_trp_brd6', 'out_trp_brd5', 'out_trp_brd4', 'out_trp_brd3','out_trp_brd2','out_trp_brd1','out_trp_brd0'))


#toa  = np.array(( 6,  64, 122, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
#surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869)) #SURFACE FOR TRP IN 'OUT' FILE

toa = 6
surf = 66

clr_trp_spc_up_boa = np.zeros(len(filename_spc))
clr_trp_spc_dn_boa = np.zeros(len(filename_spc))
clr_trp_spc_nt_boa = np.zeros(len(filename_spc))
clr_trp_brd_up_boa = np.zeros(len(filename_spc))
clr_trp_brd_dn_boa = np.zeros(len(filename_spc))
clr_trp_brd_nt_boa = np.zeros(len(filename_spc))
clr_trp_spc_up_toa = np.zeros(len(filename_spc))
clr_trp_spc_dn_toa = np.zeros(len(filename_spc))
clr_trp_spc_nt_toa = np.zeros(len(filename_spc))
clr_trp_brd_up_toa = np.zeros(len(filename_spc))
clr_trp_brd_dn_toa = np.zeros(len(filename_spc))
clr_trp_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        clr_trp_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_trp_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_trp_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_trp_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_trp_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_trp_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        clr_trp_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_trp_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_trp_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_trp_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_trp_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_trp_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

#c = np.array((89,80,70,60,50,40,30,25,22,20,18,15,10,5,0))
c = np.array((89,80,70,60,50,40,30,20,10,0))
diff_net_boa = clr_trp_spc_nt_boa - clr_trp_brd_nt_boa
diff_net_toa = clr_trp_spc_nt_toa - clr_trp_brd_nt_toa
diff_net_boa = np.flip(diff_net_boa)
diff_net_toa = np.flip(diff_net_toa)
c = np.flip(c)
diff_net_atm = diff_net_toa - diff_net_boa
spc_alb = np.flip(clr_trp_spc_up_boa / clr_trp_spc_dn_boa)
spc_abs = 1 - spc_alb
brd_alb = np.flip(clr_trp_brd_up_boa / clr_trp_brd_dn_boa)
brd_abs = 1 - brd_alb

diff_abs = spc_abs - brd_abs

nt_spc = np.flip(clr_trp_spc_dn_boa -  clr_trp_spc_up_boa)
nt_brd = np.flip(clr_trp_brd_dn_boa -  clr_trp_brd_up_boa)
nt_flx = nt_spc - nt_brd
#plt.plot(c , np.flip(clr_trp_spc_up_boa / clr_trp_spc_dn_boa), label='spc')
#plt.plot(c , np.flip(clr_trp_brd_up_boa / clr_trp_brd_dn_boa), label='brd')
diff_alb = np.flip(clr_trp_spc_up_boa / clr_trp_spc_dn_boa) - np.flip(clr_trp_brd_up_boa / clr_trp_brd_dn_boa)
#plt.plot(c , diff_alb* np.flip(clr_trp_spc_dn_toa), label='diff')
#plt.legend()
#plt.show()
fig, axs = plt.subplots(1, 3, figsize=(18,5))
axs[0].axhline(y=0.0, color='k', linestyle='-')
axs[1].axhline(y=0.0, color='k', linestyle='-')
axs[2].axhline(y=0.0, color='k', linestyle='-')

axs[0].plot(c, diff_net_boa, marker='o', label='BOA')
#axs[0,0].plot(c, diff_net_toa, marker='o', label='TOA')
axs[0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[0].plot(c, diff_net_toa, marker='o', label='TOA')
#axs[0].plot(c ,( (diff_alb)* np.flip(clr_trp_spc_dn_toa)), label='diff', marker='x')
#axs[0].plot(c, ( (diff_abs)* np.flip(clr_trp_spc_dn_toa)), label='diff', marker='x')
#axs[0].plot(c, nt_flx, label='diff', marker='x')
axs[0].set_title('Solar Zenith Angle')
axs[0].set(xlabel='Solar Zenith Angle', ylabel='Change in Net flux [W/m$^2$]')


#diff_net_boa = diff_net_boa/np.cos(np.deg2rad(c))
#diff_net_toa = diff_net_toa/np.cos(np.deg2rad(c))
#plt.plot(c, diff_net_boa, marker='o', label='BOA')
#plt.plot(c, diff_net_toa, marker='o', label='TOA')
#plt.xlabel('Solar Zenith Angle')
#plt.ylabel('Change in Net flux [W/m$^2$]')
#plt.title('Cosine Weighted Change in Net Flux vs Solar Zenith Angle')
#plt.legend()
#plt.show()


### CLOUD OPTICAL DEPTH ###

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/wtr_tau/'
filename_brd = np.array(('out_trp_brd0', 'out_trp_brdpt1', 'out_trp_brdpt2', 'out_trp_brdpt3', 'out_trp_brdpt4', 'out_trp_brdpt5', 'out_trp_brdpt7', 'out_trp_brd1', 'out_trp_brd1pt5', 'out_trp_brd2', 'out_trp_brd2pt5', 'out_trp_brd3','out_trp_brd4', 'out_trp_brd6', 'out_trp_brd8', 'out_trp_brd10', 'out_trp_brd15','out_trp_brd25','out_trp_brd35','out_trp_brd55','out_trp_brd75','out_trp_brd100'))
filename_spc = np.array(('out_trp_spc0', 'out_trp_spcpt1', 'out_trp_spcpt2', 'out_trp_spcpt3', 'out_trp_spcpt4', 'out_trp_spcpt5', 'out_trp_spcpt7', 'out_trp_spc1', 'out_trp_spc1pt5', 'out_trp_spc2', 'out_trp_spc2pt5', 'out_trp_spc3','out_trp_spc4', 'out_trp_spc6', 'out_trp_spc8', 'out_trp_spc10', 'out_trp_spc15','out_trp_spc25','out_trp_spc35','out_trp_spc55','out_trp_spc75','out_trp_spc100'))

cld_trp_spc_up_boa = np.zeros(len(filename_spc))
cld_trp_spc_dn_boa = np.zeros(len(filename_spc))
cld_trp_spc_nt_boa = np.zeros(len(filename_spc))
cld_trp_brd_up_boa = np.zeros(len(filename_spc))
cld_trp_brd_dn_boa = np.zeros(len(filename_spc))
cld_trp_brd_nt_boa = np.zeros(len(filename_spc))

cld_trp_spc_up_toa = np.zeros(len(filename_spc))
cld_trp_spc_dn_toa = np.zeros(len(filename_spc))
cld_trp_spc_nt_toa = np.zeros(len(filename_spc))
cld_trp_brd_up_toa = np.zeros(len(filename_spc))
cld_trp_brd_dn_toa = np.zeros(len(filename_spc))
cld_trp_brd_nt_toa = np.zeros(len(filename_spc))

for j in range( len(filename_spc) ) :
        cld_trp_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        cld_trp_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0,15.0,25.0,35.0,55.0,75.0,100))
diff_net_boa = cld_trp_spc_nt_boa - cld_trp_brd_nt_boa
diff_net_toa = cld_trp_spc_nt_toa - cld_trp_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa
axs[1].set_xscale('log')
#axs[0,1].set_xscale('symlog')
axs[1].plot(c[1:], diff_net_boa[1:], marker='o', label='BOA')
axs[1].plot(c[1:], diff_net_atm[1:], marker='o', label='ATM')
axs[1].plot(c[1:], diff_net_toa[1:], marker='o', label='TOA')
axs[1].set_title('Cloud Optical Depth')
axs[1].set(xlabel='Cloud Optical Depth', ylabel='Change in Net flux [W/m$^2$]')
#print(diff_net_boa)
#print((diff_net_boa / cld_trp_brd_nt_boa ) *100)

'''
### Wtr Grain  Radius  ###

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/wtr_rds/'
filename_brd = np.array(('out_trp_brd1','out_trp_brd2','out_trp_brd3','out_trp_brd4','out_trp_brd5','out_trp_brd6','out_trp_brd7','out_trp_brd8','out_trp_brd9','out_trp_brd10'))
filename_spc = np.array(('out_trp_spc1','out_trp_spc2','out_trp_spc3','out_trp_spc4','out_trp_spc5','out_trp_spc6','out_trp_spc7','out_trp_spc8','out_trp_spc9','out_trp_spc10'))

cld_trp_spc_up_boa = np.zeros(len(filename_spc))
cld_trp_spc_dn_boa = np.zeros(len(filename_spc))
cld_trp_spc_nt_boa = np.zeros(len(filename_spc))
cld_trp_brd_up_boa = np.zeros(len(filename_spc))
cld_trp_brd_dn_boa = np.zeros(len(filename_spc))
cld_trp_brd_nt_boa = np.zeros(len(filename_spc))

cld_trp_spc_up_toa = np.zeros(len(filename_spc))
cld_trp_spc_dn_toa = np.zeros(len(filename_spc))
cld_trp_spc_nt_toa = np.zeros(len(filename_spc))
cld_trp_brd_up_toa = np.zeros(len(filename_spc))
cld_trp_brd_dn_toa = np.zeros(len(filename_spc))
cld_trp_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        cld_trp_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        
        cld_trp_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((100,200,300,400,500,600,700,800,900,1000))
diff_net_boa = cld_trp_spc_nt_boa - cld_trp_brd_nt_boa
diff_net_toa = cld_trp_spc_nt_toa - cld_trp_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa

axs[1,0].plot(c, diff_net_boa, marker='o', label='BOA')
axs[1,0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[1,0].set_title('Effective Snow Grain Radius')
axs[1,0].set(xlabel='Effective Snow Grain Radius [$\mu m$]', ylabel='Change in Net flux [W/m$^2$]')
'''

### Water Vapor Concetration ###
path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/wtr_wvc/'
#filename_brd = np.array(('out_trp_brd00','out_trp_brd01','out_trp_brd02','out_trp_brd03','out_trp_brd04','out_trp_brd05','out_trp_brd06','out_trp_brd07','out_trp_brd08','out_trp_brd09','out_trp_brd1','out_trp_brd2','out_trp_brd3','out_trp_brd4','out_trp_brd5','out_trp_brd6','out_trp_brd7','out_trp_brd8','out_trp_brd9','out_trp_brd100'))
#filename_spc = np.array(('out_trp_spc00','out_trp_spc01','out_trp_spc02','out_trp_spc03','out_trp_spc04','out_trp_spc05','out_trp_spc06','out_trp_spc07','out_trp_spc08','out_trp_spc09','out_trp_spc1','out_trp_spc2','out_trp_spc3','out_trp_spc4','out_trp_spc5','out_trp_spc6','out_trp_spc7','out_trp_spc8','out_trp_spc9','out_trp_spc100'))
c = np.array((0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100))
filename_brd = np.array(('out_trp_brd1','out_trp_brd2','out_trp_brd3','out_trp_brd4','out_trp_brd5','out_trp_brd6','out_trp_brd7','out_trp_brd8','out_trp_brd9','out_trp_brd100'))
filename_spc = np.array(('out_trp_spc1','out_trp_spc2','out_trp_spc3','out_trp_spc4','out_trp_spc5','out_trp_spc6','out_trp_spc7','out_trp_spc8','out_trp_spc9','out_trp_spc100'))
c = np.array((10,20,30,40,50,60,70,80,90,100))


cld_trp_spc_up_boa = np.zeros(len(filename_spc))
cld_trp_spc_dn_boa = np.zeros(len(filename_spc))
cld_trp_spc_nt_boa = np.zeros(len(filename_spc))
cld_trp_brd_up_boa = np.zeros(len(filename_spc))
cld_trp_brd_dn_boa = np.zeros(len(filename_spc))
cld_trp_brd_nt_boa = np.zeros(len(filename_spc))

cld_trp_spc_up_toa = np.zeros(len(filename_spc))
cld_trp_spc_dn_toa = np.zeros(len(filename_spc))
cld_trp_spc_nt_toa = np.zeros(len(filename_spc))
cld_trp_brd_up_toa = np.zeros(len(filename_spc))
cld_trp_brd_dn_toa = np.zeros(len(filename_spc))
cld_trp_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        cld_trp_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        cld_trp_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_trp_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_trp_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_trp_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

#c = np.array((0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100))
diff_net_boa = cld_trp_spc_nt_boa - cld_trp_brd_nt_boa
diff_net_toa = cld_trp_spc_nt_toa - cld_trp_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa


spc_atm_abs = cld_trp_spc_nt_toa - cld_trp_spc_nt_boa
spc_boa_abs = cld_trp_spc_dn_boa - cld_trp_spc_up_boa
spc_boa_alb = cld_trp_spc_up_boa / cld_trp_spc_dn_boa

print(diff_net_boa)
print((diff_net_boa / cld_trp_brd_nt_boa ) *100)
print(diff_net_atm)

axs[2].plot(c, diff_net_boa, marker='o', label='BOA')
axs[2].plot(c, diff_net_atm, marker='o', label='ATM')
axs[2].plot(c, diff_net_toa, marker='o', label='TOA')
#axs[1,1].plot(c, spc_atm_abs, marker='o', label='SPC ATM ABS')
#axs[1,1].plot(c, spc_boa_abs, marker='o', label='SPC BOA ABS')
#axs[1,1].plot(c, spc_boa_alb, marker='o', label='SPC BOA ALB')
axs[2].set_title('Water Vapor Concentration')
axs[2].set(xlabel='Column Relative Humidity [%]', ylabel='Change in Net flux [W/m$^2$]')

fig.suptitle('Water - Single Parameter Sensitivity', fontsize='16',fontweight='bold')
plt.legend()

label = ['a','b','c']
for i in range(3):
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    axs[i].text(0.9, 0.1, label[i], transform=axs[i].transAxes + trans,
    fontsize='14', verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))



#axs[0].axhline(y=0.0, color='k', linestyle='-')
#axs[1].axhline(y=0.0, color='k', linestyle='-')
#axs[2].axhline(y=0.0, color='k', linestyle='-')

axs[2].set_ylim(-0.1,0.05)
plt.savefig('/Users/jtolento/Desktop/ppr1/trp_sps.eps')
plt.show()



