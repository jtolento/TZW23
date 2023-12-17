#/bin/env python
'''
Python scrpit to generate figures 1, 2, and 3 for TZW23
'''

from matplotlib import pyplot as plt
from netCDF4 import Dataset
from os import path
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")


### --- FUNCTIONS --- ###
def add_value_labels(ax, spacing=1): # Add values to bar plots
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.2f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)

def arrange(array): # Rearranges plots from linear RRTM array scheme to linear array
    array1 = array[1:14]
    array2 = np.zeros(len(array))
    array1 = np.flip(array1)
    for i in range(len(array1)):
        array2[i] = array1[i]
    array2[13] = array[14]
    array2[14] = array[0]
    return array2

def plt_hr(ttl, atm_prf, hr_spc_clr, hr_brd_clr, hr_spc_cld, hr_brd_cld, p_clr, p_cld): # Plot clear and cloudy atmospheric heating rates
    mxrw_clr = len(p_clr)
    mxrw_cld = len(p_cld)
    plt.plot(hr_spc_clr[29:mxrw_clr], p_clr[29:mxrw_clr], label='Clear - Spectral', color='b')
    plt.plot(hr_brd_clr[29:mxrw_clr], p_clr[29:mxrw_clr], label='Clear - Broad', color ='b', linestyle='dotted')
    plt.plot(hr_spc_cld[25:mxrw_cld], p_cld[25:mxrw_cld], label='Cloudy - Spectral', color='r')
    plt.plot(hr_brd_cld[25:mxrw_cld], p_cld[25:mxrw_cld], label='Cloudy - Broad', color ='r', linestyle='dotted')
    plt.gca().invert_yaxis()
    plt.ylim(p_cld[mxrw_cld-1], 200)
    plt.title(ttl+' Solar Warming Rate', fontweight='bold', fontsize=16)
    plt.xlabel('Warming Rate [K/day]', fontweight='bold', fontsize=14)
    plt.ylabel('Pressure [mb]', fontweight='bold', fontsize=14)
    plt.legend(prop={'size': 9})
    plt.grid( which='major', axis='both')
    plt.savefig('/Users/jtolento/Desktop/ppr1/'+atm_prf+'_hr.eps')
    plt.show()

def bias(ttl, atm_prf, data): # Bar plot of atmopsheric Biases
    if(atm_prf == 'sas'):
        location = 3
    else:
        location = 0
    labels = ['Clear', 'Cloudy']
    x = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, data[0], width/2, label='BOA')
    rects2 = ax.bar(x,           data[1], width/2, label='ATM')
    rects3 = ax.bar(x + width/2, data[2], width/2, label='TOA')
    ax.set_ylabel('Change in Flux [W m$^{-2}$]', fontweight='bold',  fontsize=14)
    ax.set_title('Change in Net Flux ('+ttl+')',  fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold',fontsize=14)
    ax.legend(loc=location)
    add_value_labels(ax)
    fig.tight_layout()
    plt.axhline(y=0.0, color='k', linestyle='-')
    plt.grid( which='major', axis='y')
    fig.savefig('/Users/jtolento/Desktop/ppr1/'+atm_prf+'_bias.eps')
    plt.show()


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

def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


### --- END FUNCTIONS --- ###


path_clr = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/clr/'
path_cld = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/cld/'


var = 7
### --- MLS --- ###
### --- CLR --- ###
filename = 'out_mls_spc'
hr_mls_spc = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(var))
p1 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(1))
dn_mls_spc = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(5))
p_mls =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(1))

filename = 'out_mls_brd'
hr_mls_brd = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(var))
p2 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(1))
dn_mls_brd = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=52, usecols=(5))


### --- CLD --- ###

mxrw = 52
diff = 0
filename = 'out_mls_spc'
hr_mls_spc_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p1_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_mls_spc_cld  = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=52, usecols=(5))
filename = 'out_mls_brd'
hr_mls_brd_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p2_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_mls_brd_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=52, usecols=(5))

#text_kwargs = dict(ha='center', va='center', fontsize=11, color='blue')
#plt.plot(dn_mls_spc[29:60], p1[29:60], label='Clear', color='b')
#plt.plot(dn_mls_spc_cld[25:mxrw], p1_cld[25:mxrw], label='Cloudy', color='r')
#plt.gca().invert_yaxis()
#plt.title('Mid-Latitude Summer Downwelling Flux', fontweight='bold')
#plt.xlabel('Downwelling Flux [W m$^{-2}$]', fontweight='bold')
#plt.ylabel('Pressure [mb]', fontweight='bold')
#plt.legend(prop={'size': 9})
#plt.savefig('/Users/jtolento/Desktop/ppr1/dwn_flx_mls.eps')
#plt.show()

net_net = np.zeros(len(hr_mls_brd_cld))
for i in range(len(hr_mls_brd_cld)-1):
    net_net[i] = hr_mls_brd_cld[i] - hr_mls_brd_cld[i+1]


#plt_hr('Mid-Latitude Summer','mls', hr_mls_spc, hr_mls_brd, hr_mls_spc_cld, hr_mls_brd_cld, p1, p1_cld)


toa  = np.array(( 6,  64, 122, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869)) #SURFACE FOR MLS IN 'OUT' FILE
clr_mls_spc_up_boa = np.zeros(len(toa))
clr_mls_spc_dn_boa = np.zeros(len(toa))
clr_mls_spc_nt_boa = np.zeros(len(toa))
clr_mls_brd_up_boa = np.zeros(len(toa))
clr_mls_brd_dn_boa = np.zeros(len(toa))
clr_mls_brd_nt_boa = np.zeros(len(toa))
clr_mls_spc_up_toa = np.zeros(len(toa))
clr_mls_spc_dn_toa = np.zeros(len(toa))
clr_mls_spc_nt_toa = np.zeros(len(toa))
clr_mls_brd_up_toa = np.zeros(len(toa))
clr_mls_brd_dn_toa = np.zeros(len(toa))
clr_mls_brd_nt_toa = np.zeros(len(toa))
cld_mls_spc_up_boa = np.zeros(len(toa))
cld_mls_spc_dn_boa = np.zeros(len(toa))
cld_mls_spc_nt_boa = np.zeros(len(toa))
cld_mls_brd_up_boa = np.zeros(len(toa))
cld_mls_brd_dn_boa = np.zeros(len(toa))
cld_mls_brd_nt_boa = np.zeros(len(toa))
cld_mls_spc_up_toa = np.zeros(len(toa))
cld_mls_spc_dn_toa = np.zeros(len(toa))
cld_mls_spc_nt_toa = np.zeros(len(toa))
cld_mls_brd_up_toa = np.zeros(len(toa))
cld_mls_brd_dn_toa = np.zeros(len(toa))
cld_mls_brd_nt_toa = np.zeros(len(toa))

for i in range( len(toa) ) :
    filename = 'out_mls_spc'
    clr_mls_spc_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_mls_spc_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_mls_spc_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_mls_spc_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_mls_spc_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_mls_spc_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_mls_brd'
    clr_mls_brd_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_mls_brd_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_mls_brd_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_mls_brd_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_mls_brd_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_mls_brd_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_mls_spc'
    cld_mls_spc_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_mls_spc_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_mls_spc_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_mls_spc_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_mls_spc_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_mls_spc_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_mls_brd'
    cld_mls_brd_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_mls_brd_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_mls_brd_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_mls_brd_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_mls_brd_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_mls_brd_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    


clr_mls_spc_up_boa = arrange(clr_mls_spc_up_boa)
clr_mls_spc_dn_boa = arrange(clr_mls_spc_dn_boa)
clr_mls_spc_nt_boa = arrange(clr_mls_spc_nt_boa)
clr_mls_spc_up_toa = arrange(clr_mls_spc_up_toa)
clr_mls_spc_dn_toa = arrange(clr_mls_spc_dn_toa)
clr_mls_spc_nt_toa = arrange(clr_mls_spc_nt_toa)
cld_mls_spc_up_boa = arrange(cld_mls_spc_up_boa)
cld_mls_spc_dn_boa = arrange(cld_mls_spc_dn_boa)
cld_mls_spc_nt_boa = arrange(cld_mls_spc_nt_boa)
cld_mls_spc_up_toa = arrange(cld_mls_spc_up_toa)
cld_mls_spc_dn_toa = arrange(cld_mls_spc_dn_toa)
cld_mls_spc_nt_toa = arrange(cld_mls_spc_nt_toa)

clr_mls_brd_up_boa = arrange(clr_mls_brd_up_boa)
clr_mls_brd_dn_boa = arrange(clr_mls_brd_dn_boa)
clr_mls_brd_nt_boa = arrange(clr_mls_brd_nt_boa)
clr_mls_brd_up_toa = arrange(clr_mls_brd_up_toa)
clr_mls_brd_dn_toa = arrange(clr_mls_brd_dn_toa)
clr_mls_brd_nt_toa = arrange(clr_mls_brd_nt_toa)
cld_mls_brd_up_boa = arrange(cld_mls_brd_up_boa)
cld_mls_brd_dn_boa = arrange(cld_mls_brd_dn_boa)
cld_mls_brd_nt_boa = arrange(cld_mls_brd_nt_boa)
cld_mls_brd_up_toa = arrange(cld_mls_brd_up_toa)
cld_mls_brd_dn_toa = arrange(cld_mls_brd_dn_toa)
cld_mls_brd_nt_toa = arrange(cld_mls_brd_nt_toa)

clr_mls_diff_up_boa = clr_mls_spc_up_boa - clr_mls_brd_up_boa
clr_mls_diff_dn_boa = clr_mls_spc_dn_boa - clr_mls_brd_dn_boa
clr_mls_diff_nt_boa = clr_mls_spc_nt_boa - clr_mls_brd_nt_boa
clr_mls_diff_up_toa = clr_mls_spc_up_toa - clr_mls_brd_up_toa
clr_mls_diff_dn_toa = clr_mls_spc_dn_toa - clr_mls_brd_dn_toa
clr_mls_diff_nt_toa = clr_mls_spc_nt_toa - clr_mls_brd_nt_toa
cld_mls_diff_up_boa = cld_mls_spc_up_boa - cld_mls_brd_up_boa
cld_mls_diff_dn_boa = cld_mls_spc_dn_boa - cld_mls_brd_dn_boa
cld_mls_diff_nt_boa = cld_mls_spc_nt_boa - cld_mls_brd_nt_boa
cld_mls_diff_up_toa = cld_mls_spc_up_toa - cld_mls_brd_up_toa
cld_mls_diff_dn_toa = cld_mls_spc_dn_toa - cld_mls_brd_dn_toa
cld_mls_diff_nt_toa = cld_mls_spc_nt_toa - cld_mls_brd_nt_toa



nt = [clr_mls_diff_nt_boa[14], cld_mls_diff_nt_boa[14]]
nt = np.array(nt)
width = 0.25
nt_boa = [clr_mls_diff_nt_boa[14], cld_mls_diff_nt_boa[14]]
nt_boa = np.array(nt_boa)
nt_toa = [clr_mls_diff_nt_toa[14], cld_mls_diff_nt_toa[14]]
nt_toa = np.array(nt_toa)
nt_atm = nt_toa - nt_boa
nt_atm = np.array(nt_atm)
data = [nt_boa, nt_atm, nt_toa]

#bias('Mid-Latitude Summer', 'mls', data)



### --- SAS --- ###
### --- CLR --- ###

mxrw = 52
toa  = np.array(( 6,  64, 112, 180, 238, 296, 354, 412, 470, 528, 586, 644, 702, 760, 818))
surf = np.array((57, 115, 173, 231, 289, 347, 405, 463, 521, 579, 637, 695, 753, 811, 869))

filename = 'out_sas_spc'
hr_sas_spc = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p1 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_sas_spc = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(5))

filename = 'out_sas_brd'
hr_sas_brd = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p2 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_sas_brd = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(5))
### --- CLD --- ###


filename = 'out_sas_spc'
hr_sas_spc_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p1_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_sas_spc_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(5))

filename = 'out_sas_brd'
hr_sas_brd_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p2_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
dn_sas_brd_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(5))


#plt.plot(dn_sas_spc[29:mxrw], p1[29:mxrw], label='Clear', color='b')
#plt.plot(dn_sas_spc_cld[29:60], p1_cld[29:60], label='Cloudy', color='r')
#plt.gca().invert_yaxis()
#plt.title('Sub-Arctic Summer Downwelling Flux', fontweight='bold')
#plt.xlabel('Downwelling Flux [W m$^{-2}$]', fontweight='bold')
#plt.ylabel('Pressure [mb]', fontweight='bold')
#plt.legend(prop={'size': 9})
#plt.savefig('/Users/jtolento/Desktop/ppr1/dwn_flx_sas.eps')
#plt.show()


#plt_hr('Sub-Arctic Summer','sas', hr_sas_spc, hr_sas_brd, hr_sas_spc_cld, hr_sas_brd_cld, p1, p1_cld)


clr_sas_spc_up_boa = np.zeros(len(toa))
clr_sas_spc_dn_boa = np.zeros(len(toa))
clr_sas_spc_nt_boa = np.zeros(len(toa))
clr_sas_brd_up_boa = np.zeros(len(toa))
clr_sas_brd_dn_boa = np.zeros(len(toa))
clr_sas_brd_nt_boa = np.zeros(len(toa))
clr_sas_spc_up_toa = np.zeros(len(toa))
clr_sas_spc_dn_toa = np.zeros(len(toa))
clr_sas_spc_nt_toa = np.zeros(len(toa))
clr_sas_brd_up_toa = np.zeros(len(toa))
clr_sas_brd_dn_toa = np.zeros(len(toa))
clr_sas_brd_nt_toa = np.zeros(len(toa))
cld_sas_spc_up_boa = np.zeros(len(toa))
cld_sas_spc_dn_boa = np.zeros(len(toa))
cld_sas_spc_nt_boa = np.zeros(len(toa))
cld_sas_brd_up_boa = np.zeros(len(toa))
cld_sas_brd_dn_boa = np.zeros(len(toa))
cld_sas_brd_nt_boa = np.zeros(len(toa))
cld_sas_spc_up_toa = np.zeros(len(toa))
cld_sas_spc_dn_toa = np.zeros(len(toa))
cld_sas_spc_nt_toa = np.zeros(len(toa))
cld_sas_brd_up_toa = np.zeros(len(toa))
cld_sas_brd_dn_toa = np.zeros(len(toa))
cld_sas_brd_nt_toa = np.zeros(len(toa))


for i in range( len(toa) ) :
    filename = 'out_sas_spc'
    clr_sas_spc_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_sas_spc_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_sas_spc_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_sas_spc_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_sas_spc_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_sas_spc_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_sas_brd'
    clr_sas_brd_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_sas_brd_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_sas_brd_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_sas_brd_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_sas_brd_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_sas_brd_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_sas_spc'
    cld_sas_spc_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_sas_spc_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_sas_spc_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_sas_spc_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_sas_spc_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_sas_spc_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_sas_brd'
    cld_sas_brd_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_sas_brd_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_sas_brd_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_sas_brd_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_sas_brd_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_sas_brd_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))

clr_sas_spc_up_boa = arrange(clr_sas_spc_up_boa)
clr_sas_spc_dn_boa = arrange(clr_sas_spc_dn_boa)
clr_sas_spc_nt_boa = arrange(clr_sas_spc_nt_boa)
clr_sas_spc_up_toa = arrange(clr_sas_spc_up_toa)
clr_sas_spc_dn_toa = arrange(clr_sas_spc_dn_toa)
clr_sas_spc_nt_toa = arrange(clr_sas_spc_nt_toa)
cld_sas_spc_up_boa = arrange(cld_sas_spc_up_boa)
cld_sas_spc_dn_boa = arrange(cld_sas_spc_dn_boa)
cld_sas_spc_nt_boa = arrange(cld_sas_spc_nt_boa)
cld_sas_spc_up_toa = arrange(cld_sas_spc_up_toa)
cld_sas_spc_dn_toa = arrange(cld_sas_spc_dn_toa)
cld_sas_spc_nt_toa = arrange(cld_sas_spc_nt_toa)

clr_sas_brd_up_boa = arrange(clr_sas_brd_up_boa)
clr_sas_brd_dn_boa = arrange(clr_sas_brd_dn_boa)
clr_sas_brd_nt_boa = arrange(clr_sas_brd_nt_boa)
clr_sas_brd_up_toa = arrange(clr_sas_brd_up_toa)
clr_sas_brd_dn_toa = arrange(clr_sas_brd_dn_toa)
clr_sas_brd_nt_toa = arrange(clr_sas_brd_nt_toa)
cld_sas_brd_up_boa = arrange(cld_sas_brd_up_boa)
cld_sas_brd_dn_boa = arrange(cld_sas_brd_dn_boa)
cld_sas_brd_nt_boa = arrange(cld_sas_brd_nt_boa)
cld_sas_brd_up_toa = arrange(cld_sas_brd_up_toa)
cld_sas_brd_dn_toa = arrange(cld_sas_brd_dn_toa)
cld_sas_brd_nt_toa = arrange(cld_sas_brd_nt_toa)

clr_sas_diff_up_boa = clr_sas_spc_up_boa - clr_sas_brd_up_boa
clr_sas_diff_dn_boa = clr_sas_spc_dn_boa - clr_sas_brd_dn_boa
clr_sas_diff_nt_boa = clr_sas_spc_nt_boa - clr_sas_brd_nt_boa
clr_sas_diff_up_toa = clr_sas_spc_up_toa - clr_sas_brd_up_toa
clr_sas_diff_dn_toa = clr_sas_spc_dn_toa - clr_sas_brd_dn_toa
clr_sas_diff_nt_toa = clr_sas_spc_nt_toa - clr_sas_brd_nt_toa
cld_sas_diff_up_boa = cld_sas_spc_up_boa - cld_sas_brd_up_boa
cld_sas_diff_dn_boa = cld_sas_spc_dn_boa - cld_sas_brd_dn_boa
cld_sas_diff_nt_boa = cld_sas_spc_nt_boa - cld_sas_brd_nt_boa
cld_sas_diff_up_toa = cld_sas_spc_up_toa - cld_sas_brd_up_toa
cld_sas_diff_dn_toa = cld_sas_spc_dn_toa - cld_sas_brd_dn_toa
cld_sas_diff_nt_toa = cld_sas_spc_nt_toa - cld_sas_brd_nt_toa


nt = [clr_sas_diff_nt_boa[14], cld_sas_diff_nt_boa[14]]
nt = np.array(nt)
width = 0.25
nt_boa = [clr_sas_diff_nt_boa[14], cld_sas_diff_nt_boa[14]]
nt_boa = np.array(nt_boa)
nt_toa = [clr_sas_diff_nt_toa[14], cld_sas_diff_nt_toa[14]]
nt_toa = np.array(nt_toa)
nt_atm = nt_toa - nt_boa
nt_atm = np.array(nt_atm)
data = [nt_boa, nt_atm, nt_toa]

#bias('Sub-Arctic Summer', 'sas', data)
    



### --- TRP --- ###
### --- CLR --- ###
mxrw = 61
filename = 'out_trp_spc'
hr_trp_spc = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p1 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
p_trp =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(1))
filename = 'out_trp_brd'
hr_trp_brd = np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p2 =  np.loadtxt(path_clr+filename, skiprows= 5, max_rows=mxrw, usecols=(1))

### --- CLD --- ###
filename = 'out_trp_spc'
hr_trp_spc_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p1_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))

filename = 'out_trp_brd'
hr_trp_brd_cld = np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(var))
p2_cld =  np.loadtxt(path_cld+filename, skiprows= 5, max_rows=mxrw, usecols=(1))

#plt_hr('Tropical','trp', hr_trp_spc, hr_trp_brd, hr_trp_spc_cld, hr_trp_brd_cld, p1, p1_cld)



toa  = np.array((6,   73, 140, 207, 274, 341, 408, 475, 542, 609, 676, 743, 810, 877,  944)) # TOA FOR TROP ATM IN 'OUT' FILE
surf = np.array((66, 133, 200, 267, 334, 401, 468, 535, 602, 669, 736, 803, 870, 937, 1004)) # SURFACE FOR TROP ATM IN 'OUT' FILE

clr_trp_spc_up_boa = np.zeros(len(toa))
clr_trp_spc_dn_boa = np.zeros(len(toa))
clr_trp_spc_nt_boa = np.zeros(len(toa))
clr_trp_brd_up_boa = np.zeros(len(toa))
clr_trp_brd_dn_boa = np.zeros(len(toa))
clr_trp_brd_nt_boa = np.zeros(len(toa))
clr_trp_spc_up_toa = np.zeros(len(toa))
clr_trp_spc_dn_toa = np.zeros(len(toa))
clr_trp_spc_nt_toa = np.zeros(len(toa))
clr_trp_brd_up_toa = np.zeros(len(toa))
clr_trp_brd_dn_toa = np.zeros(len(toa))
clr_trp_brd_nt_toa = np.zeros(len(toa))
cld_trp_spc_up_boa = np.zeros(len(toa))
cld_trp_spc_dn_boa = np.zeros(len(toa))
cld_trp_spc_nt_boa = np.zeros(len(toa))
cld_trp_brd_up_boa = np.zeros(len(toa))
cld_trp_brd_dn_boa = np.zeros(len(toa))
cld_trp_brd_nt_boa = np.zeros(len(toa))
cld_trp_spc_up_toa = np.zeros(len(toa))
cld_trp_spc_dn_toa = np.zeros(len(toa))
cld_trp_spc_nt_toa = np.zeros(len(toa))
cld_trp_brd_up_toa = np.zeros(len(toa))
cld_trp_brd_dn_toa = np.zeros(len(toa))
cld_trp_brd_nt_toa = np.zeros(len(toa))

for i in range( len(toa) ) :
    filename = 'out_trp_spc'
    clr_trp_spc_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_trp_spc_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_trp_spc_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_trp_spc_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_trp_spc_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_trp_spc_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_trp_brd'
    clr_trp_brd_up_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    clr_trp_brd_dn_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    clr_trp_brd_nt_boa[i] = np.loadtxt(path_clr+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    clr_trp_brd_up_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    clr_trp_brd_dn_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    clr_trp_brd_nt_toa[i] = np.loadtxt(path_clr+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_trp_spc'
    cld_trp_spc_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_trp_spc_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_trp_spc_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_trp_spc_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_trp_spc_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_trp_spc_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))
    filename = 'out_trp_brd'
    cld_trp_brd_up_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(2))
    cld_trp_brd_dn_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(5))
    cld_trp_brd_nt_boa[i] = np.loadtxt(path_cld+filename, skiprows = surf[i]-1, max_rows=1, usecols=(6))
    cld_trp_brd_up_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(2))
    cld_trp_brd_dn_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(5))
    cld_trp_brd_nt_toa[i] = np.loadtxt(path_cld+filename, skiprows = toa[i]-1, max_rows=1, usecols=(6))


clr_trp_spc_up_boa = arrange(clr_trp_spc_up_boa)
clr_trp_spc_dn_boa = arrange(clr_trp_spc_dn_boa)
clr_trp_spc_nt_boa = arrange(clr_trp_spc_nt_boa)
clr_trp_spc_up_toa = arrange(clr_trp_spc_up_toa)
clr_trp_spc_dn_toa = arrange(clr_trp_spc_dn_toa)
clr_trp_spc_nt_toa = arrange(clr_trp_spc_nt_toa)
cld_trp_spc_up_boa = arrange(cld_trp_spc_up_boa)
cld_trp_spc_dn_boa = arrange(cld_trp_spc_dn_boa)
cld_trp_spc_nt_boa = arrange(cld_trp_spc_nt_boa)
cld_trp_spc_up_toa = arrange(cld_trp_spc_up_toa)
cld_trp_spc_dn_toa = arrange(cld_trp_spc_dn_toa)
cld_trp_spc_nt_toa = arrange(cld_trp_spc_nt_toa)

clr_trp_brd_up_boa = arrange(clr_trp_brd_up_boa)
clr_trp_brd_dn_boa = arrange(clr_trp_brd_dn_boa)
clr_trp_brd_nt_boa = arrange(clr_trp_brd_nt_boa)
clr_trp_brd_up_toa = arrange(clr_trp_brd_up_toa)
clr_trp_brd_dn_toa = arrange(clr_trp_brd_dn_toa)
clr_trp_brd_nt_toa = arrange(clr_trp_brd_nt_toa)
cld_trp_brd_up_boa = arrange(cld_trp_brd_up_boa)
cld_trp_brd_dn_boa = arrange(cld_trp_brd_dn_boa)
cld_trp_brd_nt_boa = arrange(cld_trp_brd_nt_boa)
cld_trp_brd_up_toa = arrange(cld_trp_brd_up_toa)
cld_trp_brd_dn_toa = arrange(cld_trp_brd_dn_toa)
cld_trp_brd_nt_toa = arrange(cld_trp_brd_nt_toa)

clr_trp_diff_up_boa = clr_trp_spc_up_boa - clr_trp_brd_up_boa
clr_trp_diff_dn_boa = clr_trp_spc_dn_boa - clr_trp_brd_dn_boa
clr_trp_diff_nt_boa = clr_trp_spc_nt_boa - clr_trp_brd_nt_boa
clr_trp_diff_up_toa = clr_trp_spc_up_toa - clr_trp_brd_up_toa
clr_trp_diff_dn_toa = clr_trp_spc_dn_toa - clr_trp_brd_dn_toa
clr_trp_diff_nt_toa = clr_trp_spc_nt_toa - clr_trp_brd_nt_toa
cld_trp_diff_up_boa = cld_trp_spc_up_boa - cld_trp_brd_up_boa
cld_trp_diff_dn_boa = cld_trp_spc_dn_boa - cld_trp_brd_dn_boa
cld_trp_diff_nt_boa = cld_trp_spc_nt_boa - cld_trp_brd_nt_boa
cld_trp_diff_up_toa = cld_trp_spc_up_toa - cld_trp_brd_up_toa
cld_trp_diff_dn_toa = cld_trp_spc_dn_toa - cld_trp_brd_dn_toa
cld_trp_diff_nt_toa = cld_trp_spc_nt_toa - cld_trp_brd_nt_toa

nt = [clr_trp_diff_nt_boa[14], cld_trp_diff_nt_boa[14]]
nt = np.array(nt)
width = 0.25
nt_boa = [clr_trp_diff_nt_boa[14], cld_trp_diff_nt_boa[14]]
nt_boa = np.array(nt_boa)
nt_toa = [clr_trp_diff_nt_toa[14], cld_trp_diff_nt_toa[14]]
nt_toa = np.array(nt_toa)
nt_atm = nt_toa - nt_boa
nt_atm = np.array(nt_atm)
data = [nt_boa, nt_atm, nt_toa]
#bias('Tropical', 'trp', data)


#location = 0
labels = ['Snow', 'Ice', 'Water']
boa = np.array((clr_mls_diff_nt_boa[14], clr_sas_diff_nt_boa[14], clr_trp_diff_nt_boa[14]))
#atm = np.array((clr_mls_diff_nt_atm[14], clr_sas_diff_nt_atm[14], clr_trp_diff_nt_atm[14]))
toa = np.array((clr_mls_diff_nt_toa[14], clr_sas_diff_nt_toa[14], clr_trp_diff_nt_toa[14]))
atm = toa - boa
x = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, boa, width/3, label='BOA')
rects2 = ax.bar(x,           atm, width/3, label='ATM')
rects3 = ax.bar(x + width/3, toa, width/3, label='TOA')
ax.set_ylabel('Change in Flux [W m$^{-2}$]', fontweight='bold',  fontsize=14)
ax.set_title('Change in Net Flux',  fontweight='bold', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold',fontsize=14)
#ax.legend(loc=location)
add_value_labels(ax)
ax.legend()
fig.tight_layout()
plt.axhline(y=0.0, color='k', linestyle='-')
plt.grid( which='major', axis='y')
fig.savefig('/Users/jtolento/Desktop/ppr1/base_bias.eps')
plt.show()

#### percent change ####
print("MLS")
mls_boa_per = clr_mls_diff_nt_boa / clr_mls_brd_nt_boa * 100
print('MLS BOA Percent change ')
print(mls_boa_per[14])


mls_toa_per = clr_mls_diff_nt_toa / clr_mls_brd_nt_toa * 100
print('MLS TOA Percent change ')
print(mls_toa_per[14])

mls_atm_per = ((clr_mls_spc_nt_toa - clr_mls_spc_nt_boa) - (clr_mls_brd_nt_toa - clr_mls_brd_nt_boa))
print("atm = ")
print(mls_atm_per[14])
mls_atm_per = mls_atm_per / (clr_mls_brd_nt_toa - clr_mls_brd_nt_boa) *100
#mls_atm_per = clr_mls_diff_nt_atm / clr_mls_brd_nt_atm * 100
print("MLS ATM Percent change")
print(mls_atm_per[14])

print("SAS")
sas_boa_per = clr_sas_diff_nt_boa / clr_sas_brd_nt_boa * 100
print('SAS BOA Percent change ')
print(sas_boa_per[14])
sas_toa_per = clr_sas_diff_nt_toa / clr_sas_brd_nt_toa * 100
print('SAS TOA Percent change ')
print(sas_toa_per[14])

sas_atm_per = ((clr_sas_spc_nt_toa - clr_sas_spc_nt_boa) - (clr_sas_brd_nt_toa - clr_sas_brd_nt_boa))
print("atm = ")
print(sas_atm_per[14])
sas_atm_per = sas_atm_per / (clr_sas_brd_nt_toa - clr_sas_brd_nt_boa) *100
#sas_atm_per = clr_sas_diff_nt_atm / clr_sas_brd_nt_atm * 100                                                                    
print("SAS ATM Percent change")
print(sas_atm_per[14])


print("TRP")
trp_boa_per = clr_trp_diff_nt_boa / clr_trp_brd_nt_boa * 100
print('TRP BOA Percent change ')
print(trp_boa_per[14])
trp_toa_per = clr_trp_diff_nt_toa / clr_trp_brd_nt_toa * 100
print('TRP TOA Percent change ')
print(trp_toa_per[14])



mxrw_mls = 52
mxrw_trp = 62
hr_mls = hr_mls_spc - hr_mls_brd
hr_sas = hr_sas_spc - hr_sas_brd
hr_trp = hr_trp_spc - hr_trp_brd


fig, axs = plt.subplots(1, 2)
axs[1].plot(hr_mls[29:mxrw_mls], p_mls[29:mxrw_mls], label='Snow', color='b')
axs[0].plot(hr_mls_spc[29:mxrw_mls], p_mls[29:mxrw_mls], color='b')
axs[0].plot(hr_mls_brd[29:mxrw_mls], p_mls[29:mxrw_mls], color='b', linestyle='--')

axs[1].plot(hr_sas[29:mxrw_mls], p_mls[29:mxrw_mls], label='Ice', color='r')
axs[0].plot(hr_sas_spc[29:mxrw_mls], p_mls[29:mxrw_mls], color='r')
axs[0].plot(hr_sas_brd[29:mxrw_mls], p_mls[29:mxrw_mls], color='r', linestyle='--')

axs[1].plot(hr_trp[25:mxrw_trp], p_trp[25:mxrw_trp], label='Water', color='g')
axs[0].plot(hr_trp_spc[25:mxrw_trp], p_trp[25:mxrw_trp], color='g')
axs[0].plot(hr_trp_brd[25:mxrw_trp], p_trp[25:mxrw_trp], color='g', linestyle='--')

axs[0].invert_yaxis()
axs[0].set_ylim(p_mls[mxrw_mls-1], 200)
axs[0].set_title('Solar Warming Rate', fontweight='bold', fontsize=14)
axs[0].set_xlabel('Warming Rate [K/day]', fontweight='bold', fontsize=14)
axs[0].set_ylabel('Pressure [mb]', fontweight='bold', fontsize=14)
#axs[0].legend(prop={'size': 9})
axs[0].grid( which='major', axis='both')

axs[1].invert_yaxis()
axs[1].set_ylim(p_mls[mxrw_mls-1], 200)
axs[1].set_title('Change', fontweight='bold', fontsize=14)
axs[1].set_xlabel('Warming Rate [K/day]', fontweight='bold', fontsize=14)
#axs[1].set_ylabel('Pressure [mb]', fontweight='bold', fontsize=14)
axs[1].legend(prop={'size': 9})
axs[1].grid( which='major', axis='both')

fig.suptitle('Change in Solar Warming Rate', fontweight='bold', fontsize='16')
plt.savefig('/Users/jtolento/Desktop/ppr1/base_hr.eps')
plt.show()




clr_mls_spc_alb = clr_mls_spc_up_boa / clr_mls_spc_dn_boa
clr_mls_spc_alb = clr_mls_spc_alb[0:14]
clr_mls_brd_alb = clr_mls_brd_up_boa / clr_mls_brd_dn_boa
clr_mls_brd_alb= clr_mls_brd_alb[0:14]
clr_sas_spc_alb = clr_sas_spc_up_boa / clr_sas_spc_dn_boa
clr_sas_spc_alb = clr_sas_spc_alb[0:14]
clr_sas_brd_alb = clr_sas_brd_up_boa / clr_sas_brd_dn_boa
clr_sas_brd_alb= clr_sas_brd_alb[0:14]
clr_trp_spc_alb = clr_trp_spc_up_boa / clr_trp_spc_dn_boa
clr_trp_spc_alb = clr_trp_spc_alb[0:14]
clr_trp_brd_alb = clr_trp_brd_up_boa / clr_trp_brd_dn_boa
clr_trp_brd_alb= clr_trp_brd_alb[0:14]


clr_mls_spc_up = clr_mls_spc_up_boa
clr_mls_spc_up = clr_mls_spc_up[0:14]
clr_mls_brd_up = clr_mls_brd_up_boa
clr_mls_brd_up = clr_mls_brd_up[0:14]
clr_sas_spc_up = clr_sas_spc_up_boa
clr_sas_spc_up = clr_sas_spc_up[0:14]
clr_sas_brd_up = clr_sas_brd_up_boa
clr_sas_brd_up = clr_sas_brd_up[0:14]
clr_trp_spc_up = clr_trp_spc_up_boa
clr_trp_spc_up = clr_trp_spc_up[0:14]
clr_trp_brd_up = clr_trp_brd_up_boa
clr_trp_brd_up = clr_trp_brd_up[0:14]
clr_mls_chg_up = clr_mls_spc_up - clr_mls_brd_up
clr_sas_chg_up = clr_sas_spc_up - clr_sas_brd_up
clr_trp_chg_up = clr_trp_spc_up - clr_trp_brd_up

# Arrays A and B with 64-bit float values
A = [200, 263.15789474, 344.82758621, 441.50110375, 625, 778.21011673, 1242.23602484, 1298.7012987, 1626.01626016, 1941.7475, 2150.53763441, 2500, 3076.92307692, 3846.15384615, 12195.12195]


# Generate x and y values for the plot
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
    spc_values_range = [clr_mls_spc_up[i]] * num_points
    brd_values_range = [clr_mls_brd_up[i]] * num_points
    chg_mls_values_range = [clr_mls_chg_up[i]] * num_points
    chg_sas_values_range = [clr_sas_chg_up[i]] * num_points
    chg_trp_values_range = [clr_trp_chg_up[i]] * num_points
    spc_alb_mls_values_range = [clr_mls_spc_alb[i]] * num_points 
    spc_alb_sas_values_range = [clr_sas_spc_alb[i]] * num_points
    spc_alb_trp_values_range = [clr_trp_spc_alb[i]] * num_points
    brd_alb_mls_values_range = [clr_mls_brd_alb[i]] * num_points
    brd_alb_sas_values_range = [clr_sas_brd_alb[i]] * num_points
    brd_alb_trp_values_range = [clr_trp_brd_alb[i]] * num_points
    
    x_values.extend(x_values_range)
    spc_values.extend(spc_values_range)
    brd_values.extend(brd_values_range)
    chg_mls_values.extend(chg_mls_values_range)
    chg_sas_values.extend(chg_sas_values_range)
    chg_trp_values.extend(chg_trp_values_range)
    spc_alb_mls_values.extend(spc_alb_mls_values_range)
    spc_alb_sas_values.extend(spc_alb_sas_values_range)
    spc_alb_trp_values.extend(spc_alb_trp_values_range)
    brd_alb_mls_values.extend(brd_alb_mls_values_range)
    brd_alb_sas_values.extend(brd_alb_sas_values_range)
    brd_alb_trp_values.extend(brd_alb_trp_values_range)

    
# Adding the last value of clr_mls_spc_up for the last range in A
num_points_last_range = int(np.ceil(A[-1] - A[-2]))
x_values_last_range = np.linspace(A[-2], A[-1], num=num_points_last_range, endpoint=False)
spc_values_last_range = [clr_mls_spc_up[-1]] * num_points_last_range
brd_values_last_range = [clr_mls_brd_up[-1]] * num_points_last_range
chg_mls_values_last_range = [clr_mls_chg_up[-1]] * num_points_last_range
chg_sas_values_last_range = [clr_sas_chg_up[-1]] * num_points_last_range
chg_trp_values_last_range = [clr_trp_chg_up[-1]] * num_points_last_range
spc_alb_mls_values_last_range = [clr_mls_spc_alb[-1]] * num_points_last_range
spc_alb_sas_values_last_range = [clr_sas_spc_alb[-1]] * num_points_last_range
spc_alb_trp_values_last_range = [clr_trp_spc_alb[-1]] * num_points_last_range
brd_alb_mls_values_last_range = [clr_mls_brd_alb[-1]] * num_points_last_range
brd_alb_sas_values_last_range = [clr_sas_brd_alb[-1]] * num_points_last_range
brd_alb_trp_values_last_range = [clr_trp_brd_alb[-1]] * num_points_last_range


x_values.extend(x_values_last_range)
spc_values.extend(spc_values_last_range)
brd_values.extend(brd_values_last_range)
chg_mls_values.extend(chg_mls_values_last_range)
chg_sas_values.extend(chg_sas_values_last_range)
chg_trp_values.extend(chg_trp_values_last_range)
spc_alb_mls_values.extend(spc_alb_mls_values_last_range)
spc_alb_sas_values.extend(spc_alb_sas_values_last_range)
spc_alb_trp_values.extend(spc_alb_trp_values_last_range)
brd_alb_mls_values.extend(brd_alb_mls_values_last_range)
brd_alb_sas_values.extend(brd_alb_sas_values_last_range)
brd_alb_trp_values.extend(brd_alb_trp_values_last_range)

# Plotting
A = [round(x / 1000, 2) for x in A]
B = A[::2]
x_values = [x / 1000 for x in x_values]

fig, axs = plt.subplots(2, 1, figsize = (12,12))
axs[0].semilogx(x_values, spc_alb_mls_values, label='Snow', color='b')
axs[0].semilogx(x_values, brd_alb_mls_values,linestyle='--', color='b')
axs[0].semilogx(x_values, spc_alb_sas_values, label='Bare-Ice', color='r')
axs[0].semilogx(x_values, brd_alb_sas_values,linestyle='--', color='r')
axs[0].semilogx(x_values, spc_alb_trp_values, label='Water', color='g')
axs[0].semilogx(x_values, brd_alb_trp_values,linestyle='--', color='g')
axs[0].set_title('Spectral vs Semi-Broadband Albedo', fontsize=16)
axs[0].set_xlabel('Wavelength [$\mu$m]', fontsize=16)
axs[0].set_ylabel('Albedo', fontsize=16)
axs[0].set_xticks(ticks=B, labels=B, minor=False)
axs[0].legend()
axs[0].grid(which='major', axis='both')


#plt.semilogx(x_values, spc_values, label='Spectral', color='b')
#plt.semilogx(x_values, brd_values, label='Semi-broadband', linestyle='--', color='b')
#plt.xlabel('Wavelength [$\mu$m]', fontsize=14)
#plt.ylabel('Albedo', fontsize=14)
#plt.title('Plot with Boundary Points', fontweight='bold', fontsize=16)
#plt.xticks(ticks=B, labels=B, minor=False)
#plt.grid( which='major', axis='both')
#plt.legend()
#plt.show()


axs[1].semilogx(x_values, chg_mls_values, label='Snow', color='b')
axs[1].semilogx(x_values, chg_sas_values, label='Bare-Ice', color='r')
axs[1].semilogx(x_values, chg_trp_values, label='Water', color='g')
axs[1].set_title('Change in Reflected Flux', fontsize=16)
axs[1].set_xlabel('Wavelength [$\mu$m]', fontsize=16)
axs[1].set_ylabel('Flux [W m$^{-2}$]', fontsize=16)
axs[1].set_xticks(ticks=B, labels=B, minor=False)
axs[1].legend()
axs[1].grid(which='major', axis='both')
fig.suptitle('Spectral vs Semi-Broadband Albedo', fontweight='bold', fontsize='16')
plt.savefig('/Users/jtolento/Desktop/ppr1/base_spc_alb_up.eps')
plt.show()







