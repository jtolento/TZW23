% Driver for SNICARv3 or SNICAR-ADv3 subroutine
clear; close all; clc; 
    
% 1= Direct-beam incident flux, 0= Diffuse incident flux
% NOTE that cloudy-sky spectral fluxes are loaded when direct_beam=0
input_args.direct_beam   = 1;   
sza = 73;
input_args.coszen = cos(deg2rad(sza));

% SNOW LAYER THICKNESSES [m]:
input_args.dz = [3.0]; % multi layer column 
%input_args.dz = [1000]; % single, optically-semi-infinite layer

nbr_lyr = length(input_args.dz);  % number of snow layers

% LAYER MEDIUM TYPE [ 1=snow, 2=ice]
%  Must have same length as dz
input_args.lyr_typ(1:nbr_lyr) = [2]; % snow, ice, ice 

% SNOW DENSITY FOR EACH LAYER (units: kg/m3)
%input_args.rho_snw(1:nbr_lyr) = [400, 650, 850]; 
%input_args.rho_snw(1:nbr_lyr) = [900]; 
% SNOW GRAIN SIZE FOR EACH LAYER (units: microns):
%input_args.rds_snw(1:nbr_lyr) = [500, 150, 300]; % snow grains, air bubbles, air bubbles 
 input_args.rds_snw(1:nbr_lyr) = [300];
% ICE REFRACTIVE INDEX DATASET TO USE:
% 1=Warren (1984); 2=Warren and Brandt (2008); 3=Picard et al. (2016); 4=CO2 ice
input_args.ice_ri = 3;
    
% Snow grain shape option
% 1=sphere; 2=spheroid; 3=hexagonal plate; 4=koch snowflake
input_args.sno_shp(1:nbr_lyr)  = 3;

% Shape factor: ratio of nonspherical grain effective radii to that of equal-volume sphere
% 0=use recommended default value (He et al. 2017);
% others(0<fs<1)= use user-specified value
% only activated when sno_shp > 1 (i.e. nonspherical)
input_args.sno_fs(1:nbr_lyr)   = 0;

% Aspect ratio: ratio of grain width to length
% 0=use recommended default value (He et al. 2017);
% others(0.1<fs<20)= use user-specified value
% only activated when sno_shp > 1 (i.e. nonspherical)
input_args.sno_ar(1:nbr_lyr)   = 0;

% type of dust: 1=Sahara, 2=Colorado, 3=Greenland, 4=Mars
input_args.dust_type = 1; 

% type of volcanic ash: 1 = Eyjafjallajokull
input_args.ash_type = 1;

use_ssl = 0;
if (use_ssl == 1)
    nbr_lyr = nbr_lyr+1;
end
% PARTICLE MASS MIXING RATIOS (ng/g or ug/g, depending on species)
% NOTE: This is mass of impurity per mass of snow
%  (i.e., mass of impurity / mass of ice+impurity)
input_args.mss_cnc_sot1(1:nbr_lyr)  = 0.0;    % uncoated black carbon [ng/g]
input_args.mss_cnc_sot2(1:nbr_lyr)  = 0.0;    % sulfate-coated black carbon [ng/g]
input_args.mss_cnc_brc1(1:nbr_lyr)  = 0.0;    % uncoated brown carbon [ng/g]
input_args.mss_cnc_brc2(1:nbr_lyr)  = 0.0;    % sulfate-coated brown carbon [ng/g]

input_args.mss_cnc_dst1(1:nbr_lyr)  = 0.0;    % dust size 1 (r=0.05-0.5um) [ug/g]
input_args.mss_cnc_dst2(1:nbr_lyr)  = 0.0;    % dust size 2 (r=0.5-1.25um) [ug/g]
input_args.mss_cnc_dst3(1:nbr_lyr)  = 0.0;    % dust size 3 (r=1.25-2.5um) [ug/g]
input_args.mss_cnc_dst4(1:nbr_lyr)  = 0.0;    % dust size 4 (r=2.5-5.0um)  [ug/g]
input_args.mss_cnc_dst5(1:nbr_lyr)  = 0.0;    % dust size 5 (r=5.0-50um)   [ug/g]

input_args.mss_cnc_ash1(1:nbr_lyr)  = 0.0;    % volcanic ash size 1 (r=0.05-0.5um) [ug/g]
input_args.mss_cnc_ash2(1:nbr_lyr)  = 0.0;    % volcanic ash size 2 (r=0.5-1.25um) [ug/g]
input_args.mss_cnc_ash3(1:nbr_lyr)  = 0.0;    % volcanic ash size 3 (r=1.25-2.5um) [ug/g]
input_args.mss_cnc_ash4(1:nbr_lyr)  = 0.0;    % volcanic ash size 4 (r=2.5-5.0um)  [ug/g]
input_args.mss_cnc_ash5(1:nbr_lyr)  = 0.0;    % volcanic ash size 5 (r=5.0-50um)   [ug/g]

input_args.snw_alg_cell_nbr_conc(1:nbr_lyr) = 0.0;    % SNOW algae [UNITS: cells/mL]
input_args.alg_rds(1:nbr_lyr)               = 10;     % mean algae cell radius (um)
input_args.dcmf_pig_chla(1:nbr_lyr)         = 0.015;  % dry cell mass fraction of chlorophyll-a
input_args.dcmf_pig_chlb(1:nbr_lyr)         = 0.005;  % dry cell mass fraction of chlorophyll-b
input_args.dcmf_pig_cara(1:nbr_lyr)         = 0.05;   % dry cell mass fraction of photoprotective_carotenoids
input_args.dcmf_pig_carb(1:nbr_lyr)         = 0.00;   % dry cell mass fraction of photosynthetic_carotenoids  

input_args.glc_alg_mss_cnc(1:nbr_lyr)       = 0.0;  % GLACIER algae [UNITS ng/g] = ppb 
input_args.glc_alg_rds                      = 4;    % GLACIER algae radius [um]
input_args.glc_alg_len                      = 40;   % GLACIER algae length [um]    
input_args.chl_wtr(1:nbr_lyr)                         = [0.0]; % [mg/m^3] Chlorophyl concentration in water

% REFLECTANCE OF SURFACE UNDERLYING SNOW: (Value is applied to all
% wavelengths. User can alternatively specify spectrally-dependent
% ground albedo in snicar_v3.m)
input_args.R_sfc_all_wvl = 0.25;

% ATMOSPHERIC PROFILE for surface-incident flux:
%     1 = mid-latitude winter
%     2 = mid-latitude summer
%     3 = sub-Arctic winter
%     4 = sub-Arctic summer
%     5 = Summit,Greenland (sub-Arctic summer, surface pressure of 796hPa)
%     6 = High Mountain (summer, surface pressure of 556 hPa)
%     7 = Top-of-atmosphere
% NOTE that clear-sky spectral fluxes are loaded when direct_beam=1,
% and cloudy-sky spectral fluxes are loaded when direct_beam=0
input_args.atm = 4;

% Broadband surface-incident solar flux [W/m2]:
%  (used to scale spectral flux fractions)
input_args.flx_dwn_bb = 1.0;

%Chlorophyll Concentraction in water layer %JPT
%input_args.chl_wtr = [0.0]; % [mg/m^3]


% COSINE OF SOLAR ZENITH ANGLE FOR DIRECT-BEAM RT
%sza = linspace(89, 21.5, 10);

input_args.rho_snw(1:nbr_lyr) = [100]; 
di1 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [200]; 
di2 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [300]; 
di3 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [400]; 
di4 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [500]; 
di5 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [600]; 
di6 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [700]; 
di7 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [800]; 
di8 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [900]; 
di9 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [916.999]; 
di10 = snicarAD_v5(input_args);

if (1==1)
    figure(1)
    plot(di1.wvl,di1.albedo,'linewidth',3, 'DisplayName','R = 100');
    hold on;
    plot(di2.wvl,di2.albedo,'linewidth',3, 'DisplayName','R = 200');
    hold on;
    plot(di3.wvl,di3.albedo,'linewidth',3, 'DisplayName','R = 300');
    hold on;
    plot(di4.wvl,di4.albedo,'linewidth',3, 'DisplayName','R = 400');
    hold on;
    plot(di5.wvl,di5.albedo,'linewidth',3, 'DisplayName','R = 500');
    hold on;
    plot(di6.wvl,di6.albedo,'linewidth',3, 'DisplayName','R = 600');
    hold on;
    plot(di7.wvl,di7.albedo,'linewidth',3, 'DisplayName','R = 700');
    hold on;
    plot(di8.wvl,di8.albedo,'linewidth',3, 'DisplayName','R = 800');
    hold on;
    plot(di9.wvl,di9.albedo,'linewidth',3, 'DisplayName','R = 900');
    hold on;
    plot(di10.wvl,di10.albedo,'linewidth',3, 'DisplayName','R = 1000');
    hold on;
    axis([0.2 2.6 0 1]);
    set(gca,'xtick',0.2:0.5:5,'fontsize',14)
    set(gca,'ytick',0:0.1:1.0,'fontsize',14);
    title('Albedo of Snow', 'fontsize',20)
    xlabel('Wavelength (\mum)','fontsize',20);
    ylabel('Hemispheric Albedo','fontsize',20);
    %text(1.9,0.8,str, 'fontsize',16)
    %annotation('textbox',dim,'String',str,'FitBoxToText','on', 'Fontsize',14);
    grid on;
    legend('fontsize',16);
    %saveas(gcf,'~/Desktop/ppr1/alb_snw','epsc')
end    %hold off;




% CALL SNICAR WITH INPUT ARGUMENTS
%di = snicarAD_v5(input_args);

%str = {'\theta = 63.6^{\circ}','\rho = 250 kg/m^3','{\itr}_e = 100 \mum'};
%dim = [0.695 0.39 0.29 0.4];
%input_args.direct_beam   = 0;   
%di2 = snicarAD_v4(input_args);



% plot modeled spectral albedo:
if (0==1)
    figure(1)
    %plot(di.wvl,di.albedo,'linewidth',3, 'DisplayName','Clear');
    hold on
    axis([0.2 2.6 0 1]);
    set(gca,'xtick',0.2:0.5:5,'fontsize',14)
    set(gca,'ytick',0:0.1:1.0,'fontsize',14);
    title('Albedo of Ice', 'fontsize',20)
    xlabel('Wavelength (\mum)','fontsize',20);
    ylabel('Hemispheric Albedo','fontsize',20);
    %text(1.9,0.8,str, 'fontsize',16)
    %annotation('textbox',dim,'String',str,'FitBoxToText','on', 'Fontsize',14);
    grid on;
    legend('fontsize',16);
    %saveas(gcf,'~/Desktop/ppr1/alb_snw','epsc')
    %hold off;
end

if (1==0)
    figure(2)
    %plot(di.wvl,di.flx_dwn_spc,'linewidth',3, 'DisplayName','Clear');
    hold on
    %plot(di.wvl,di2.flx_dwn_spc,'linewidth',3, 'DisplayName','Cloudy');
    %axis([0.2 2.6 0 1]);
    %set(gca,'xtick',0.2:0.5:5,'fontsize',14)
    %set(gca,'ytick',0:0.1:1.0,'fontsize',14);
    title('Mid-Latitude Summer Insolation', 'fontsize',20)
    xlabel('Wavelength (\mum)','fontsize',20);
    ylabel('Fractional Insolation','fontsize',20);
    text(1.9,0.8,str, 'fontsize',16)
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'Fontsize',14);
    grid on;
    legend('fontsize',16);
    hold on 
    %saveas(gcf,'~/Desktop/ppr1/mls_insolation','epsc')
end

[spc, brd] = binning(di1);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc1';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd1';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di2);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc2';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd2';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di3);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc3';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd3';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di4);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc4';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd4';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di5);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc5';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd5';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di6);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc6';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd6';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di7);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc7';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd7';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di8);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc8';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd8';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di9);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc9';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd9';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


[spc, brd] = binning(di10);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc10';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd10';
line = 7;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line)


input_args.direct_beam   = 0;  
input_args.rho_snw(1:nbr_lyr) = [100]; 
di1 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [200]; 
di2 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [300]; 
di3 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [400]; 
di4 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [500]; 
di5 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [600]; 
di6 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [700]; 
di7 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [800]; 
di8 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [900]; 
di9 = snicarAD_v5(input_args);
input_args.rho_snw(1:nbr_lyr) = [916.999]; 
di10 = snicarAD_v5(input_args);



[spc, brd] = binning(di1);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc1';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd1';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di2);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc2';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd2';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di3);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc3';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd3';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di4);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc4';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd4';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di5);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc5';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd5';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di6);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc6';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd6';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di7);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc7';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd7';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);


[spc, brd] = binning(di8);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc8';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd8';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di9);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc9';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd9';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);

[spc, brd] = binning(di10);
spc_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_spc10';  % Replace with the actual file path
brd_path = '/Users/jtolento/TZW23/RRTMG_SW/run_examples_std_atm/ppr1/sps/ice_rho/input_sas_brd10';
line = 8;          %7 for direct, 8 for diffuse
modify_rrtm_input(spc_path, spc,line);
modify_rrtm_input(brd_path, brd,line);