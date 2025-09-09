function Fij = getFij(Xi, Xj, sigma, eps_K)
%getFij.m computes the interatomic force between two atoms 
% using the Lennard-Jones (12-6) potential.
%
% Inputs:
%   Xi     : (1x3) position vector of atom i (X,Y,Z) 
%   Xj     : (1x3) position vector of atom j (X,Y,Z)
%   sigma  : Lennard-Jones zero-crossing distance (m)
%   eps_K  : Lennard-Jones well-depth (K)
%
% Outputs:
%   Fij    : (1x3) force vector in (N)

    drij = norm(Xi-Xj);   % Interatomic distance (m)
    
    drijxy = norm(Xi(1:2)-Xj(1:2));             % Distance in XY-plane (m)
    thetaij = atan2(Xj(3)-Xi(3) , drijxy);      % Angle between XY- and Z-plane (rad)
    phiij   = atan2(Xj(2)-Xi(2) , Xj(1)-Xi(1)); % Angle between X and Y axes (rad)

    Fij_norm = LJ_force(drij, sigma, eps_K);

    % Decompose the force in (x,y,z) directions
    Fijz  = sin(thetaij)*Fij_norm; 
    Fijxy = cos(thetaij)*Fij_norm;
    Fijx  = cos(phiij)*Fijxy; 
    Fijy  = sin(phiij)*Fijxy;
    
    Fij = [-Fijx, -Fijy, -Fijz];   % (N)
end
