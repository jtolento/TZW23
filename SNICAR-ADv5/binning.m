function [spc,brd] = binning(di)
    
    % Practice binning to set the spectrally resolved broadband alb in RRTM - Split band that straddles vis-nir interface
    % Define wvl and bins
    wavenumber = [820, 2600, 3250, 4000, 4650, 5150, 6150, 7700, 8050, 12850, 16000, 22650, 29000, 38000, 50000]; % wavenumber in [cm-1] 820-2600
    wvln = 1e7 ./ wavenumber; % wvl of the bands in nm
    wvln = fliplr(wvln) / 1e3;

    % Primariny input required is alb
    alb = di.albedo; 
    frc = di.flx_dwn_spc;
    frc = frc / sum(frc);
    wvl = di.wvl;
   
    
    puny = 1e-10;
    frc(frc < puny) = puny;
    num_bins = 14;
    % Generate bin edges
    bin_edges = wvln;
    % Initialize variables
    binned_alb = zeros(1, num_bins);
    counts = zeros(1, num_bins);
    
    % Perform weighted binning
    for i = 1:length(alb)
        wl = wvl(i);
        weight = frc(i);
        bin_index = find(wl >= bin_edges, 1, 'last');
        
        if ~isempty(bin_index)
            binned_alb(bin_index) = binned_alb(bin_index) + alb(i) * weight;
            counts(bin_index) = counts(bin_index) + weight;
        end
    end
    
    % Compute weighted average for each bin
    for i = 1:num_bins
        if counts(i) > 0
            binned_alb(i) = binned_alb(i) / counts(i);
        end
    end
    
    %%%% Broadband %%%%%%%
    
    % Generate bin edges
    bin_edges = [0.2 0.7 12.19];
    % Initialize variables
    num_bins = 2;
    binned_alb_brd = zeros(1, num_bins);
    counts = zeros(1, num_bins);
    
    % Perform broadband weighted binning
    for i = 1:length(alb)
        wl = wvl(i);
        weight = frc(i);
        bin_index = find(wl >= bin_edges, 1, 'last');
        
        if ~isempty(bin_index)
            binned_alb_brd(bin_index) = binned_alb_brd(bin_index) + alb(i) * weight;
            counts(bin_index) = counts(bin_index) + weight;
        end
    end
    
    % Compute weighted average for each bin
    for i = 1:num_bins
        if counts(i) > 0
            binned_alb_brd(i) = binned_alb_brd(i) / counts(i);
        end
    end
    
    mid_splt = 0.0;
    low_splt = 0.0;
    for i = 1:length(alb)
        if (and(wvl(i) >= 0.6250, wvl(i) <= 0.7782))
            mid_splt = mid_splt+frc(i);
        end
        if (and(wvl(i) >= 0.6250, wvl(i) <= 0.7))
           low_splt = low_splt + frc(i);
        end
    end
        
   low_splt = low_splt / mid_splt;
   hi_splt  = 1 - low_splt;

   binned_alb_brd = [binned_alb_brd(1) binned_alb_brd(1) binned_alb_brd(1) binned_alb_brd(1) ...
       binned_alb_brd(1)*low_splt + binned_alb_brd(2)*hi_splt ...
       binned_alb_brd(2) binned_alb_brd(2)  binned_alb_brd(2)  binned_alb_brd(2)  binned_alb_brd(2)  binned_alb_brd(2)  binned_alb_brd(2)  binned_alb_brd(2) binned_alb_brd(2)]; 
    % Display the binned alb values
    %disp(binned_alb);
    %disp(binned_alb_brd)
    
    binned_alb =  1- fliplr(binned_alb);
    binned_alb = [binned_alb(2:14) binned_alb(1)];

   binned_alb_brd = 1 - fliplr(binned_alb_brd);
   binned_alb_brd = [binned_alb_brd(2:14) binned_alb_brd(1)]; 
   spc = binned_alb;
   brd = binned_alb_brd;
   
end

