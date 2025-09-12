function energy = LJ_energy(r, sigma, eps_K)

    K_to_ev = 0.00008617328149741;
    ev_to_J = 1.60217662e-19;

    eps = eps_K*K_to_ev*ev_to_J;  % Joule

    energy = 4*eps*((sigma./r).^12 - (sigma./r).^6);
end
