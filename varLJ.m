function [force] = varLJ(r,sigma_LJ,eps_LJ)
%LJh.m computes the interatomic force according to the Lennard-Jones (12-6)
%function with the parameters sigma and epsilon.
%
% Inputs:
%   r  : The interatomic distance 
%
% Outputs:
%   force : The interatomic force

    K_to_ev = 0.00008617328149741;      % Kelvin to ev conversion
    ev_to_J = 1.60217662*10^(-19);      % ev to Joule conversion
    
    rA = r;                             % Intermolecular distance (m)
    sigma = sigma_LJ;                 % LJ zero-crossing distance (m)
    eps = eps_LJ;                        % Well-depth (K)
    eps = eps*K_to_ev;                  % Well-depth (eV)
    eps = eps*ev_to_J;                  % Well-depth (J)
       
    % LJ Force in (N)
    force = -4*eps*((6*sigma^6./rA.^7)-(12*sigma^12./rA.^13));
end

