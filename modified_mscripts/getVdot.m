function Vdot = getVdot(F,m)
%getVdot.m computes the acceleration of a rigid body according to Newton's
%second law.
%
% Inputs:
%   F  : Total force acting on a body 
%   m  : Mass of the body
%
% Outputs:
%   Vdot, M2 : The acceleration of the body. In 1D this is a scalar, and in
%   multi-dimensional system, the acceleration has the same shape as the
%   force F.

% Acceleration in m/s^2
Vdot = F/m;

end

