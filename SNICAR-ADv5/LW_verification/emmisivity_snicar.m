%clear all; close all; clc;
% e_vs_theta_v8
% effective angle at 10 microns: 50.82

% 100*[cm-1] = [m-1]
min_bound_wvnm=100.*[10.0, 350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0, ...
			     1180.0, 1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2390.0, 2600.0]; 
max_bound_wvnm=100.*[350.0, 500.0, 630.0, 700.0, 820.0, 980.0,1080.0,1180.0, ...
			     1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2390.0, 2600.0, 3250.0];

% convert from wavenumber to wavelength (1/wvnm) in microns (*1e6), 
% flip from decreasing to increasing value
min_bound_wvl = flip(1e6./max_bound_wvnm); % switch min and max since inverting! [micrometers]
max_bound_wvl = flip(1e6./min_bound_wvnm); % [micrometers]
n1 = 1; % air

n_rl = readtable('foo_nr.txt'); % load data 
n_rl =table2array(n_rl);
n_im = readtable('foo_ni.txt'); %
n_im = table2array(n_im);
wvl = readtable('foo_wvl.txt');  % [micrometers]
wvl = table2array(wvl);
%begin = 1;
%last = 10;
%plot(wvl(begin:end),n_rl(begin:end));
%xlim([0 15])

lwr_bnd = min_bound_wvl(1);
upr_bnd = max_bound_wvl(16);
j = 0;
for i=1:length(wvl)
   if wvl(i) <= lwr_bnd
      j = j+1; 
   end
end
k=0;
for i=1:length(wvl)
   if wvl(i) >= upr_bnd
      k = k+1; 
   end
end

wvl = wvl(465:end-360);
nr = n_rl(465:end-360);
ni = n_im(465:end-360);
wvl_len = length(wvl);

%%
% Changing variable to match SNICAR inputs
rfidx_re = zeros(length(nr),1);
rfidx_re(:,1) = nr;
rfidx_im = ni;
k=1;
coszen = cos(53*pi/180);
mu_not= coszen;
mu0 = mu_not;





temp1 = rfidx_re.^2 - rfidx_im.^2 +sin(acos(coszen)).^2;
temp2 = rfidx_re.^2 - rfidx_im.^2 -sin(acos(coszen)).^2;
Nreal = (sqrt(2)/2) .* ( temp1 + (temp2.^2 + 4*rfidx_re.^2.*rfidx_im.^2).^(0.5) ).^0.5;

nr = Nreal;
mu0n = cos(asin(sin(acos(mu0))/nr));
nr_array = nr;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SNICAR CODE COPY AND PASTED %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 for iw=1:wvl_len
                nr = nr_array(iw);
                mu0n = cos(asin(sin(acos(mu0))/nr));
                % ice complex index of refraction 
                refindx = complex(rfidx_re(iw, k), rfidx_im(iw));
                % critical angle where total internal reflection occurs 
                critical_angle = asin(refindx);
                
                % compare incoming angle to critical angle 
                if acos(mu_not) < critical_angle
                    
                %! compute fresnel reflection and transmission amplitudes
                %! for two polarizations: 1=perpendicular and 2=parallel to
                %! the plane containing incident, reflected and refracted rays.
                    
                    %! Eq. (5.4.18a-b); Liou 2002
                    R1 = (mu0-nr*mu0n) / (mu0 + nr*mu0n);      %reflection amplitude factor for perpendicular polarization
                    R2 = (nr*mu0 - mu0n) / (nr*mu0 + mu0n);    %reflection amplitude factor for parallel polarization
                    T1 = 2*mu0 / (mu0 + nr*mu0n);              %transmission amplitude factor for perpendicular polarization
                    T2 = 2*mu0 / (nr*mu0 + mu0n);              %transmission amplitude factor for parallel polarization
                    
                    %! unpolarized light for direct beam
                    %! Eq. 21; Brigleb and light 2007
                    Rf_dir_a = 0.5 * ((R1^2) + (R2^2));
                    Tf_dir_a = 0.5 * (T1*T1 + T2*T2)*nr*mu0n/mu0;
               
                else % total internal reflection 
                    Tf_dir_a = 0;
                    Rf_dir_a = 1;
                    
                
                end
 
                R(iw) = Rf_dir_a;
 end
 

emiss = 1 - R;
plot(1e4./wvl, emiss, 'k', 'lineWidth',3)
xlim([0 3400])

