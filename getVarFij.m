function Fij = getVarFij(Xi, Xj, sigma_LJ, eps_LJ)
%getFij.m computes the interatomic force between two atoms with a specified
% interatomic potential.
%
% Inputs:
%   Xi  : A (1x3) or (3x1) position vector of atom i (X,Y,Z) 
%   Xj  : A (1x3) or (3x1) position vector of atom j (X,Y,Z)
%   Note that the forms of Xi and Xj must be consistent 
%
% Outputs:
%   Fij : A (1x3) vector containing the interatomic forces (Fx, Fy, Fz)

drij = norm(Xi-Xj);                         % Interatomic distance (m)
    
% Geometrical computations
drijxy = norm(Xi(1:2)-Xj(1:2));             % Interatomic distance in XY-plane (m)
thetaij = atan2(Xj(3)-Xi(3) , drijxy);      % Interatomic angle between the XY- and Z-planes (rad)
phiij = atan2(Xj(2)-Xi(2) , Xj(1)-Xi(1));   % Interatomic angle between the X- and Y planes (rad)

% Computing the net/norm of the interatomic force
Fij = varLJ(drij, sigma_LJ, eps_LJ);

% Decompose the force in the (X,Y,Z) directions
Fijz = sin(thetaij)*Fij; Fijxy = cos(thetaij)*Fij;
Fijx = cos(phiij)*Fijxy; Fijy = sin(phiij)*Fijxy;

Fij = [-Fijx, -Fijy, -Fijz];                % Force vector in (N)

end

