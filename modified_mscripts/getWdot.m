function [Wdot]= getWdot(M, I)
%getWdotH2 computes angular velocity derivatives
%   w = angular velocity vector as a (3x1) or (1x3) vector
%   M = torque vector as (3x1) or (1x3)
%   I = principal moment of inertia
% note that the shapes of w and M must be consistent with each other

% Decomposing the moment vector
M1 = M(1);
M2 = M(2);

% Computing the angular accelerations according to Newton's second law.
% Angular accelerations in (rad/s^2)

W1dot = M1/I;
W2dot = M2/I;
W3dot = 0;

% Angular acceleration vector (wx, wy, wz)
Wdot = [W1dot,W2dot,W3dot];