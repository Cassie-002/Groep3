function R = getRandRotMat
% getRandRotMat.m creates a random rotational orientation by constructing a
%   random (3x3) rotation matrix R
%
% Inputs:
%   none
%
% Outputs:
%   R   : A (3x3) rotation matrix according to the ZYX Euler angles

psi     = rand*2*pi;            % Euler angle about z-direction
theta   = 0;                    % Euler angle about y-direction
phi    = acos(1-2*rand);        % Euler angle about x-direction

Rz = [cos(psi), -sin(psi), 0; sin(psi), cos(psi), 0; 0, 0, 1];
Ry = [cos(theta), 0, sin(theta); 0, 1, 0; -sin(theta), 0, cos(theta)];
Rx = [1, 0, 0;0,cos(phi),-sin(phi);0,sin(phi),cos(phi)];
R  = Rz*Ry*Rx; 
    
end

