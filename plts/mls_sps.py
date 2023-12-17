#/bin/env python                                                                    
'''                                                                                 
Python scrpit to generate figure 4 for TZW23
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

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_sza/'
filename_spc = np.array(('out_mls_spc89', 'out_mls_spc8', 'out_mls_spc7', 'out_mls_spc6', 'out_mls_spc5', 'out_mls_spc4', 'out_mls_spc3', 'out_mls_spc2', 'out_mls_spc1', 'out_mls_spc0'))

filename_brd = np.array(('out_mls_brd89', 'out_mls_brd8', 'out_mls_brd7', 'out_mls_brd6', 'out_mls_brd5', 'out_mls_brd4', 'out_mls_brd3', 'out_mls_brd2', 'out_mls_brd1', 'out_mls_brd0'))

toa  = np.array(( 6,  64, 122, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869)) #SURFACE FOR MLS IN 'OUT' FILE

toa = 6
surf = 57

clr_mls_spc_up_boa = np.zeros(len(filename_spc))
clr_mls_spc_dn_boa = np.zeros(len(filename_spc))
clr_mls_spc_nt_boa = np.zeros(len(filename_spc))
clr_mls_brd_up_boa = np.zeros(len(filename_spc))
clr_mls_brd_dn_boa = np.zeros(len(filename_spc))
clr_mls_brd_nt_boa = np.zeros(len(filename_spc))
clr_mls_spc_up_toa = np.zeros(len(filename_spc))
clr_mls_spc_dn_toa = np.zeros(len(filename_spc))
clr_mls_spc_nt_toa = np.zeros(len(filename_spc))
clr_mls_brd_up_toa = np.zeros(len(filename_spc))
clr_mls_brd_dn_toa = np.zeros(len(filename_spc))
clr_mls_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        clr_mls_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_mls_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_mls_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_mls_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_mls_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_mls_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        clr_mls_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        clr_mls_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        clr_mls_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        clr_mls_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        clr_mls_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        clr_mls_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((89,80,70,60,50,40,30,20,10,0))
diff_net_boa = clr_mls_spc_nt_boa - clr_mls_brd_nt_boa
diff_net_toa = clr_mls_spc_nt_toa - clr_mls_brd_nt_toa
diff_net_boa = np.flip(diff_net_boa)
diff_net_toa = np.flip(diff_net_toa)


c = np.flip(c)
diff_net_atm = diff_net_toa - diff_net_boa

print('SZA')
print(diff_net_boa)
print((diff_net_boa / np.flip(clr_mls_brd_nt_boa) ) *100)
print('\n')

print(diff_net_atm)
print((diff_net_atm / np.flip((clr_mls_brd_nt_toa - clr_mls_brd_nt_boa)))*100)
print('\n')

fig, axs = plt.subplots(2, 2, figsize=(12,10))
axs[0,0].plot(c, diff_net_boa, marker='o', label='BOA')
axs[0,0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[0,0].plot(c, diff_net_toa, marker='o', label='TOA')
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

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_tau/'
filename_brd = np.array(('out_mls_brd0', 'out_mls_brd01', 'out_mls_brd02', 'out_mls_brd03', 'out_mls_brd04', 'out_mls_brd05', 'out_mls_brd06', 'out_mls_brd07', 'out_mls_brd08', 'out_mls_brd09','out_mls_brdpt1', 'out_mls_brdpt2', 'out_mls_brdpt3', 'out_mls_brdpt4', 'out_mls_brdpt5', 'out_mls_brdpt7', 'out_mls_brd1', 'out_mls_brd1pt5', 'out_mls_brd2', 'out_mls_brd2pt5', 'out_mls_brd3','out_mls_brd4', 'out_mls_brd6', 'out_mls_brd8', 'out_mls_brd10', 'out_mls_brd15','out_mls_brd25','out_mls_brd35','out_mls_brd55','out_mls_brd75'))
filename_spc = np.array(('out_mls_spc0', 'out_mls_spc01', 'out_mls_spc02', 'out_mls_spc03', 'out_mls_spc04', 'out_mls_spc05', 'out_mls_spc06', 'out_mls_spc07', 'out_mls_spc08', 'out_mls_spc09','out_mls_spcpt1', 'out_mls_spcpt2', 'out_mls_spcpt3', 'out_mls_spcpt4', 'out_mls_spcpt5', 'out_mls_spcpt7', 'out_mls_spc1', 'out_mls_spc1pt5', 'out_mls_spc2', 'out_mls_spc2pt5', 'out_mls_spc3','out_mls_spc4', 'out_mls_spc6', 'out_mls_spc8', 'out_mls_spc10', 'out_mls_spc15','out_mls_spc25','out_mls_spc35','out_mls_spc55','out_mls_spc75'))

cld_mls_spc_up_boa = np.zeros(len(filename_spc))
cld_mls_spc_dn_boa = np.zeros(len(filename_spc))
cld_mls_spc_nt_boa = np.zeros(len(filename_spc))
cld_mls_brd_up_boa = np.zeros(len(filename_spc))
cld_mls_brd_dn_boa = np.zeros(len(filename_spc))
cld_mls_brd_nt_boa = np.zeros(len(filename_spc))

cld_mls_spc_up_toa = np.zeros(len(filename_spc))
cld_mls_spc_dn_toa = np.zeros(len(filename_spc))
cld_mls_spc_nt_toa = np.zeros(len(filename_spc))
cld_mls_brd_up_toa = np.zeros(len(filename_spc))
cld_mls_brd_dn_toa = np.zeros(len(filename_spc))
cld_mls_brd_nt_toa = np.zeros(len(filename_spc))

for j in range( len(filename_spc) ) :
        cld_mls_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))

        cld_mls_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0,15.0,25.0,35.0,55.0,75.0))
diff_net_boa = cld_mls_spc_nt_boa - cld_mls_brd_nt_boa
diff_net_toa = cld_mls_spc_nt_toa - cld_mls_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa

print('TAU')
print(diff_net_boa)
print((diff_net_boa / cld_mls_brd_nt_boa ) *100)
print('\n')

print(diff_net_atm)
print((diff_net_atm / (cld_mls_brd_nt_toa - cld_mls_brd_nt_boa))*100)
print('\n')


axs[0,1].set_xscale('log')
#axs[0,1].set_xscale('symlog')
axs[0,1].plot(c[1:], diff_net_boa[1:], marker='o', label='BOA')
axs[0,1].plot(c[1:], diff_net_atm[1:], marker='o', label='ATM')
axs[0,1].plot(c[1:], diff_net_toa[1:], marker='o', label='TOA')
axs[0,1].set_title('Cloud Optical Depth')
axs[0,1].set(xlabel='Cloud Optical Depth', ylabel='Change in Net flux [W/m2]')
#print(diff_net_boa)
#print((diff_net_boa / cld_mls_brd_nt_boa ) *100)

### Snw Grain  Radius  ###

path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_rds/'
filename_brd = np.array(('out_mls_brd1','out_mls_brd2','out_mls_brd3','out_mls_brd4','out_mls_brd5','out_mls_brd6','out_mls_brd7','out_mls_brd8','out_mls_brd9','out_mls_brd10'))
filename_spc = np.array(('out_mls_spc1','out_mls_spc2','out_mls_spc3','out_mls_spc4','out_mls_spc5','out_mls_spc6','out_mls_spc7','out_mls_spc8','out_mls_spc9','out_mls_spc10'))

cld_mls_spc_up_boa = np.zeros(len(filename_spc))
cld_mls_spc_dn_boa = np.zeros(len(filename_spc))
cld_mls_spc_nt_boa = np.zeros(len(filename_spc))
cld_mls_brd_up_boa = np.zeros(len(filename_spc))
cld_mls_brd_dn_boa = np.zeros(len(filename_spc))
cld_mls_brd_nt_boa = np.zeros(len(filename_spc))

cld_mls_spc_up_toa = np.zeros(len(filename_spc))
cld_mls_spc_dn_toa = np.zeros(len(filename_spc))
cld_mls_spc_nt_toa = np.zeros(len(filename_spc))
cld_mls_brd_up_toa = np.zeros(len(filename_spc))
cld_mls_brd_dn_toa = np.zeros(len(filename_spc))
cld_mls_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        cld_mls_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        
        cld_mls_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

c = np.array((100,200,300,400,500,600,700,800,900,1000))
diff_net_boa = cld_mls_spc_nt_boa - cld_mls_brd_nt_boa
diff_net_toa = cld_mls_spc_nt_toa - cld_mls_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa
print('RDS')
print(diff_net_boa)
print((diff_net_boa / cld_mls_brd_nt_boa ) *100)
print('\n')

print(diff_net_atm)
print(diff_net_atm / (cld_mls_brd_nt_toa - cld_mls_brd_nt_boa))
print('\n')



axs[1,0].plot(c, diff_net_boa, marker='o', label='BOA')
axs[1,0].plot(c, diff_net_atm, marker='o', label='ATM')
axs[1,0].plot(c, diff_net_toa, marker='o', label='TOA')
axs[1,0].set_title('Effective Snow Grain Radius')
axs[1,0].set(xlabel='Effective Snow Grain Radius [$\mu m$]', ylabel='Change in Net flux [W/m2]')


### Water Vapor Concetration ###
path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_wvc/'
#filename_brd = np.array(('out_mls_brd00','out_mls_brd01','out_mls_brd02','out_mls_brd03','out_mls_brd04','out_mls_brd05','out_mls_brd06','out_mls_brd07','out_mls_brd08','out_mls_brd09','out_mls_brd1','out_mls_brd2','out_mls_brd3','out_mls_brd4','out_mls_brd5','out_mls_brd6','out_mls_brd7','out_mls_brd8','out_mls_brd9','out_mls_brd100'))
#filename_spc = np.array(('out_mls_spc00','out_mls_spc01','out_mls_spc02','out_mls_spc03','out_mls_spc04','out_mls_spc05','out_mls_spc06','out_mls_spc07','out_mls_spc08','out_mls_spc09','out_mls_spc1','out_mls_spc2','out_mls_spc3','out_mls_spc4','out_mls_spc5','out_mls_spc6','out_mls_spc7','out_mls_spc8','out_mls_spc9','out_mls_spc100'))
c = np.array((0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100))
filename_brd = np.array(('out_mls_brd1','out_mls_brd2','out_mls_brd3','out_mls_brd4','out_mls_brd5','out_mls_brd6','out_mls_brd7','out_mls_brd8','out_mls_brd9','out_mls_brd100'))
filename_spc = np.array(('out_mls_spc1','out_mls_spc2','out_mls_spc3','out_mls_spc4','out_mls_spc5','out_mls_spc6','out_mls_spc7','out_mls_spc8','out_mls_spc9','out_mls_spc100'))
c = np.array((10,20,30,40,50,60,70,80,90,100))


cld_mls_spc_up_boa = np.zeros(len(filename_spc))
cld_mls_spc_dn_boa = np.zeros(len(filename_spc))
cld_mls_spc_nt_boa = np.zeros(len(filename_spc))
cld_mls_brd_up_boa = np.zeros(len(filename_spc))
cld_mls_brd_dn_boa = np.zeros(len(filename_spc))
cld_mls_brd_nt_boa = np.zeros(len(filename_spc))

cld_mls_spc_up_toa = np.zeros(len(filename_spc))
cld_mls_spc_dn_toa = np.zeros(len(filename_spc))
cld_mls_spc_nt_toa = np.zeros(len(filename_spc))
cld_mls_brd_up_toa = np.zeros(len(filename_spc))
cld_mls_brd_dn_toa = np.zeros(len(filename_spc))
cld_mls_brd_nt_toa = np.zeros(len(filename_spc))
for j in range( len(filename_spc) ) :
        cld_mls_spc_up_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_boa[j] = np.loadtxt(path+filename_spc[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_spc_up_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_spc_dn_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_spc_nt_toa[j] = np.loadtxt(path+filename_spc[j], skiprows = toa-1, max_rows=1, usecols=(6))
        cld_mls_brd_up_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_boa[j] = np.loadtxt(path+filename_brd[j], skiprows = surf-1, max_rows=1, usecols=(6))
        cld_mls_brd_up_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(2))
        cld_mls_brd_dn_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(5))
        cld_mls_brd_nt_toa[j] = np.loadtxt(path+filename_brd[j], skiprows = toa-1, max_rows=1, usecols=(6))

#c = np.array((0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100))
diff_net_boa = cld_mls_spc_nt_boa - cld_mls_brd_nt_boa
diff_net_toa = cld_mls_spc_nt_toa - cld_mls_brd_nt_toa
diff_net_atm = diff_net_toa - diff_net_boa


spc_atm_abs = cld_mls_spc_nt_toa - cld_mls_spc_nt_boa
spc_boa_abs = cld_mls_spc_dn_boa - cld_mls_spc_up_boa
spc_boa_alb = cld_mls_spc_up_boa / cld_mls_spc_dn_boa
print('WVC')
print(diff_net_boa)                                                                                           
print((diff_net_boa / cld_mls_brd_nt_boa ) *100)
print('\n')
print(diff_net_atm)
print((diff_net_atm / (cld_mls_brd_nt_toa - cld_mls_brd_nt_boa))*100)
print('\n')
axs[1,1].plot(c, diff_net_boa, marker='o', label='BOA')
axs[1,1].plot(c, diff_net_atm, marker='o', label='ATM')
axs[1,1].plot(c, diff_net_toa, marker='o', label='TOA')
#axs[1,1].plot(c, spc_atm_abs, marker='o', label='SPC ATM ABS')
#axs[1,1].plot(c, spc_boa_abs, marker='o', label='SPC BOA ABS')
#axs[1,1].plot(c, spc_boa_alb, marker='o', label='SPC BOA ALB')
axs[1,1].set_title('Water Vapor Concentration')
axs[1,1].set(xlabel='Column Relative Humidity [%]', ylabel='Change in Net flux [W/m2]')

fig.suptitle('Snow - Single Parameter Sensitivity', fontsize='16',fontweight='bold')
plt.legend()

plt.savefig('/Users/jtolento/Desktop/ppr1/mls_sps.eps')
plt.show()




### WATER VAPOR INSOLATION ###
def wtr_plot(filename, humidity):
        path_clr = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/snw_wvc/'
        surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869))
        A = [200, 263.15789474, 344.82758621, 441.50110375, 625, 778.21011673, 1242.23602484, 1298.7012987, 1626.01626016, 1941.7475, 2150.53763441, 2500, 3076.92307692, 3846.15384615, 12195.12195]
        clr_mls_spc_dn_boa = np.zeros(len(surf))
        if (filename == 'snicar'):
                clr_mls_spc_dn_boa = np.array((0.0000, 0.0159,0.1181,0.2993,0.1904, 0.2483, 0.0233, 0.0419, 0.0262, 0.0112, 0.0164, 0.0003, 0.0051, 0.0035))
                humidity = 'SNICAR'
        else:
                for i in range( len(surf) ) :
                        clr_mls_spc_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
                
                max_flux = clr_mls_spc_dn_boa[0]

                humidity = np.char.mod('%s%%', humidity)

                clr_mls_spc_dn_boa = arrange(clr_mls_spc_dn_boa)
                clr_mls_spc_dn_boa = clr_mls_spc_dn_boa / max_flux
        
        x_values = []
        spc_values = []
        brd_values = []
        chg_mls_values = []
        chg_sas_values = []
        chg_trp_values = []
        
        spc_alb_mls_values = []
        spc_alb_sas_values = []
        spc_alb_trp_values = []
        brd_alb_mls_values = []
        brd_alb_sas_values = []
        brd_alb_trp_values = []

        for i in range(len(A) - 1):
                num_points = int(np.ceil(A[i + 1] - A[i]))
                x_values_range = np.linspace(A[i], A[i + 1], num=num_points, endpoint=False)
                spc_values_range = [clr_mls_spc_dn_boa[i]] * num_points
                x_values.extend(x_values_range)
                spc_values.extend(spc_values_range)
                

        num_points_last_range = int(np.ceil(A[-1] - A[-2]))
        x_values_last_range = np.linspace(A[-2], A[-1], num=num_points_last_range, endpoint=False)
        spc_values_last_range = [clr_mls_spc_dn_boa[-1]] * num_points_last_range
        

        A = [round(x / 1000, 2) for x in A]
        B = A[::2]
        x_values = [x / 1000 for x in x_values]
        plt.semilogx(x_values, spc_values, label=humidity)
        plt.xticks(ticks=B, labels=B, minor=False)


plt.figure(figsize=(9,6))
#for i in range(len(filename_spc)):
for i in range(0, 10, 2):
        wtr_plot(filename_spc[i],c[i])



wtr_plot('snicar',c[i])


plt.title('Fractional Flux vs Humidty',  fontweight='bold',  fontsize=16)
plt.xlabel('Wavelength [$\mu$m]', fontsize=14)
plt.ylabel('Fractional Flux', fontsize=14)
plt.grid(which='major', axis='both')
plt.legend()
plt.show()
