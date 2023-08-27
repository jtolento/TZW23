clear all; close all; clc;
% e_vs_theta_v8
% effective angle at 10 microns: 50.82

% 100*[cm-1] = [m-1]
min_bound_wvnm=100.*[10.0, 350.0, 500.0, 630.0, 700.0, 820.0, 980.0, 1080.0, ...
			     1180.0, 1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2390.0, 2600.0]; 
max_bound_wvnm=100.*[350.0, 500.0, 630.0, 700.0, 820.0, 980.0,1080.0,1180.0, ...
			     1390.0, 1480.0, 1800.0, 2080.0, 2250.0, 2390.0, 2600.0, 3250.0];

% convert from wavenumber to wavelength (1/wvnm) in microns (*1e6), 
% flip from decreasing to increasing value
min_bound_wvl = flip(1e6./max_bound_wvnm); % switch min and max since inverting!
max_bound_wvl = flip(1e6./min_bound_wvnm);
n1 = 1; % air

n_rl = readtable('idx_rfr_h2o_lqd_tbl_rl-Table 1.csv'); % load data
n_im = readtable('idx_rfr_h2o_lqd_tbl_img-Table 1.csv');
wvl = readtable('idx_rfr_h2o_lqd_tbl_wvl-Table 1.csv');

m_re0 = table2array(n_rl(:,2:end)); % convert to arrays and put into single row
m_im0 = table2array(n_im(:,2:end)); % 2:end don't need first column
wvl_a0 = table2array(wvl(:,2:end)); 

% sort wavelengths in ascending order and put into single row
% get values and corresponding indices
[wvl_a, w_indices] = sort(wvl_a0(:)); 


% [wvl_a_interp, w_indices_interp] = sort(wvl_array_interp); 

m_re = []; m_im = [];
for band_indices=1:length(w_indices) % match m indices with sorted wavelength indices
    m_re = [m_re m_re0(w_indices(band_indices))];
    m_im = [m_im m_im0(w_indices(band_indices))];
end
 %m_im = m_im(1:1272);
[E_band_avg,E_fill_band_a,wvl_a_found_a,save_E,E_cont] = deal([]);

theta_i_eff = 53.82*pi/180; % radians
%plot(wvl_a, m_re, 'r', 'LineWidth', 5)
%hold on;
%plot(wvl_a, m_im, 'k', 'LineWidth', 2)

% figure(1)
for h=1:length(max_bound_wvl)-0 % for each of the 16 bands
    band_indices=[]; m_re_found=[]; m_im_found=[]; wvl_a_found=[]; 

    % find indices of wavelengths within each band
    band_indices_found=find(wvl_a>=min_bound_wvl(h) & wvl_a<max_bound_wvl(h));
    band_indices=[band_indices band_indices_found];

    N_r=[];

    % get m_re, m_im, and wavelength values corresponding to wavelength indices
    for bi_length=1:length(band_indices)
        m_re_found=[m_re_found m_re(band_indices(bi_length))];
        m_im_found=[m_im_found m_im(band_indices(bi_length))];
        wvl_a_found=[wvl_a_found wvl_a(band_indices(bi_length))];

        % calculate emissivities for each value within the band at given
        % effective angle theta_eff

        % calculate N_r from m_re and m_im
        N_r =[N_r ((sqrt(2)/2) .* (m_re_found(bi_length).^2 - m_im_found(bi_length).^2 + ...
            sin(theta_i_eff).^2 + ((m_re_found(bi_length).^2 - m_im_found(bi_length).^2 - ...
            sin(theta_i_eff).^2) + 4.*m_re_found(bi_length).^2 .* m_im_found(bi_length).^2).^.5).^.5)]; 

        % calculate theta_t from effective incident angle theta_eff
        theta_t_eff = asin(sin(theta_i_eff)./N_r);

        % calculate perpendicular and parallel reflectivity
        R_perp = (cos(theta_i_eff) - N_r.*cos(theta_t_eff)) ./ ...
            (cos(theta_i_eff) + N_r.*cos(theta_t_eff));
    
        R_par = (N_r .* cos(theta_i_eff) - cos(theta_t_eff)) ./ ...
            (N_r .* cos(theta_i_eff) + cos(theta_t_eff));

        % calculate total reflectivity and then emissivity
        R = .5.*(R_par.^2+R_perp.^2);
        E = 1-R;
        
    end
    save_E=[save_E E];
    
%     figure, plot(wvl_a_found, E)

    % get average emissivity per band
    E_band_avg = [E_band_avg mean(E)];
    E_fill_band = ones(1,length(band_indices))*mean(E); % gaps created here

%     disp(mean(E))

%     plot(1e4./wvl_a_found, E_fill_band, 'r', 'LineWidth', 3), hold on % wavenumber


    % append data into single array for comparison with v3
    E_fill_band_a = cat(2, E_fill_band_a, E_fill_band); 
    wvl_a_found_a = cat(2, wvl_a_found_a, wvl_a_found);
    E_cont = cat(2, E_cont, E);
    

    figure(10)
    plot(1e4./wvl_a_found, E_fill_band, 'r', 'LineWidth', 3), hold on  % wavenumber
    plot(1e4./wvl_a_found_a, save_E, 'LineWidth', 3)
    hold on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 20230125 JPT: Verification %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbr_wvl = length(wvl_a);
coszen = theta_i_eff;
mu_not = coszen;
mu0 = mu_not;


rfidx_re = m_re;
rfidx_im = m_im;

temp1 = rfidx_re.^2 - rfidx_im.^2 +sin(acos(coszen)).^2;
temp2 = rfidx_re.^2 - rfidx_im.^2 -sin(acos(coszen)).^2;
Nreal_wtr = (sqrt(2)/2) .* ( temp1 + (temp2.^2 + 4*rfidx_re.^2.*rfidx_im.^2).^(0.5) ).^0.5;
        
nr = Nreal_wtr;
%mu0n = cos(asin(sin(acos(mu0))/nr));

for iw=1:nbr_wvl
    refindx = complex(rfidx_re(iw), rfidx_im(iw));
    critical_angle = asin(refindx);
    nr = Nreal_wtr(iw);
    mu0n = cos(asin(sin(acos(mu0))/nr));
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
                    Rf_dir_a(iw) = 0.5 * ((R1^2) + (R2^2));
                    Tf_dir_a(iw) = 0.5 * (T1*T1 + T2*T2)*nr*mu0n/mu0;
               
                else % total internal reflection 
                    Tf_dir_a(iw) = 0;
                    Rf_dir_a(iw) = 1;
                    
                end % if critical angle check 
        
end

jpt_emiss = 1 - Rf_dir_a;
plot(1e4./wvl_a(450:1200),jpt_emiss(450:1200), 'g', 'lineWidth',3)


% plot E as a constant in each band
xlabel('Wavelength (microns)'), ylabel('Emissivity'), title('Water')
set(gca, 'FontSize', 20)
grid on
% ylim([.8 1])
% xlim([-150 3400])

%% Weighted average

%h = 6.63*10^(-34);
%c = 3*10^8;
%k = 1.38 * 10^(-23);
%nu = (c*10^6) / (1e4./wvl_a);
%T = 273;
%for iw=1:nbr_wvl
%    B(iw) = ((2* h * nu(iw)^3) / c^2) * (1 / (exp((h*nu(iw))/(k*T)) - 1));
%end
%figure(11)
%plot(1e4./wvl_a, B, 'r', 'lineWidth',3)


% figure(20), plot(1e4./wvl_a_found_a, save_E, 'LineWidth', 3)
% set(gca,'FontSize',26)
% xlim([0 3500])
% hold on



% figure(20), scatter(1e4./wvl_a_found_a, E_cont, 'k', 'LineWidth', 3, 'DisplayName', 'Data'), hold on
% scatter(1e4./wvl_a_found_a, E_fill_band_a, 'r', 'LineWidth', 3, 'DisplayName', 'Band-average')
% % xlim([0 34])
% set(gca,'FontSize',20), legend
% %xlabel('Wavelength (microns)'), ylabel('Emissivity'), title('Water')
% grid on
% ylim([.8 1])
% xlim([-150 3400])












