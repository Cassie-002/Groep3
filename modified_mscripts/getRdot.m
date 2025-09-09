function Rdot = getRdot(w,R)
%getRdot.m computes the derivative of the rotation matrix R given an
%angular velocity vector w
%
% Inputs:
%   w  : Angular velocity vector of a body 
%   R  : Rotation matrix of a body
%
% Outputs:
%   Rdot : the (3x3) derivative of the rotation matrix R

% wtilde is the skew-symmetric (3x3) matrix of the angular veloctiy vector w
wtilde = [...
          0, -w(3), w(2);
          w(3), 0, -w(1);
          -w(2), w(1), 0];

% Rdot is the (3x3) derivative of the rotation matrix R
Rdot = R*wtilde;
end

