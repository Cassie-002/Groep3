function [energy] = varLJ_e(r,sigma_LJ,eps_LJ)           % Force in (N) for a given distance in (m)
%LJh.m computes the interatomic energy according to the Lennard-Jones (12-6)
%function with the parameters sigma and epsilon.
%
% Inputs:
%   r  : The interatomic distance between two atoms 
%
% Outputs:
%   energy : The interatomic potential energy 

    K_to_ev = 0.00008617328149741;      % Kelvin to ev conversion
    ev_to_J = 1.60217662*10^(-19);      % ev to Joule conversion

    rA = r;                             % Intermolecular distance (m)
    sigma = sigma_LJ;                 % LJ zero-crossing distance (m)
    eps = eps_LJ;                        % Well-depth (K)
    eps = eps*K_to_ev;                  % Well-depth (eV)
    eps = eps*ev_to_J;                  % Well-depth (J)
       
    % LJ Force in (J)
    energy = 4*eps*((sigma./rA).^12-(sigma./rA).^6);
end

