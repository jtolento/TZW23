 % nc filename to be written
file = 'water_properties.nc' ;

%% Write lon and lat variables
FL_r_dif_a_wtr = readmatrix('interp_wtr_dif_a.txt');                   %JPT
FL_r_dif_b_wtr = readmatrix('interp_wtr_dif_a.txt');                   %JPT
rfidx_wtr_re = readmatrix('interp_wtr_nr.txt');                %JPT
rfidx_wtr_im = readmatrix('interp_wtr_ni.txt'); 
atm_trop = readmatrix('atm_trop.txt');
atm_trop_cld = readmatrix('atm_trop_cld.txt');
A_chl = readmatrix('A_chl.txt');
E_chl = readmatrix('E_chl.txt');
sca_cff_wtr      = readmatrix('scatt_wtr.txt');
wvl     = [0.205:0.01:4.995];
nx = length(wvl);
nccreate(file,'wvl','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'FL_r_dif_a_wtr','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'FL_r_dif_b_wtr','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'rfidx_wtr_re','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'rfidx_wtr_im','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'atm_trop','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'atm_trop_cld','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'A_chl','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'E_chl','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;
nccreate(file,'sca_cff_wtr','Dimensions',{'wvl',1,nx},'DeflateLevel',7) ;



%ncwrite('FL_reflection_diffuse_real_may3.nc',str1, R_diffuse1)
ncwrite(file,'FL_r_dif_a_wtr',FL_r_dif_a_wtr); 
ncwrite(file,'FL_r_dif_b_wtr',FL_r_dif_b_wtr); 
ncwrite(file,'rfidx_wtr_re',rfidx_wtr_re); 
ncwrite(file,'rfidx_wtr_im',rfidx_wtr_im); 
ncwrite(file,'atm_trop',atm_trop); 
ncwrite(file,'atm_trop_cld',atm_trop_cld);
ncwrite(file,'A_chl',A_chl); 
ncwrite(file,'E_chl',E_chl);
ncwrite(file,'sca_cff_wtr',sca_cff_wtr);




%%
% lon = 1:10 ;
% lat = 1:10 ;
% nx = length(lon) ;
% nccreate(file,'lon','Dimensions',{'lon',1,nx},'DeflateLevel',7) ;
% ny = length(lat) ;
% nccreate(file,'lat','Dimensions',{'lat',1,ny},'DeflateLevel',7) ;
% nccreate(file,'time','Dimensions',{'time',1,Inf},'DeflateLevel',7) ;
% nccreate(file,'z','Dimensions',{'lon','lat','time'},'DeflateLevel',7) ;
% for i = 1:10
%     ncwrite(file,'time',i,i)   % write time 
%     data = rand(10) ; 
%     ncwrite(file,'z',data,[1,1,i]) ;   % write 3D data 
% end