function force = LJ_force(r, sigma, eps_K)
%LJ_force.m computes the interatomic force using the
% Lennard-Jones (12-6) potential.
%
% Inputs:
%   r     : Interatomic distance (m)
%   sigma : Lennard-Jones zero-crossing distance (m)
%   eps_K : Lennard-Jones well-depth (K)
%
% Outputs:
%   force : Interatomic force (N)
    K_to_ev = 0.00008617328149741;   % Kelvin to eV
    ev_to_J = 1.60217662e-19;        % eV to Joule
    eps = eps_K*K_to_ev*ev_to_J;

    force = -4*eps*((6*sigma^6 ./ r.^7) - (12*sigma^12 ./ r.^13));
end
