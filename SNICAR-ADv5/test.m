clc;
clear all;
close all;

a = svp_H2O_lqd_PrK78(290);

function svp_H2O_lqd_PrK = svp_H2O_lqd_PrK78(tpt)
    % Saturation vapor pressure over planar liquid water, PrK78
    % Compute saturation vapor pressure over planar liquid water
    % Input temperature in degrees kelvin
    % Saturation vapor pressure returned in [Pa]
    % Taken from PrK78 p. 625
    % Range of validity is -50 C < T < 50 C

    % Fundamental and derived physical constants
    tpt_frz_pnt = 273.15; % Freezing point in Kelvin

    % Constants
    cff = [6.107799961, 4.436518521e-1, 1.428945805e-2, 2.650648471e-4, ...
           3.031240396e-6, 2.034080948e-8, 6.136820929e-11];
       
    % Main code
    tpt_cls = tpt - tpt_frz_pnt; % [C]
    if tpt_cls > -50.0
        svp_H2O_lqd_dbl = cff(1) + tpt_cls * (cff(2) + tpt_cls * (cff(3) + ...
            tpt_cls * (cff(4) + tpt_cls * (cff(5) + tpt_cls * (cff(6) + ...
            cff(7) * tpt_cls)))));
        svp_H2O_lqd_PrK = svp_H2O_lqd_dbl * 100.0; % [mb] --> [Pa]
    end
    return 
end

line
