function [M1, M2] = getM(F13tr, F14tr, F23tr, F24tr, R1, R2, bondLength)
%getM.m computes the total torque (momentum vector) of two diatomic 
%molecules in their body-fixed frames given the 4 interatomic forces.
%
% Inputs:
%   Fijtr      : Interatomic force vector(s) between atoms i and j [N]
%   R1, R2     : Rotation matrices of molecules 1 and 2
%   bondLength : Distance between the 2 atoms of a diatomic molecule [m]
%
% Outputs:
%   M1, M2     : (1x3) torque vectors [N·m] in the body-fixed frame
%                for molecules 1 and 2 respectively

    % Interatomic forces rotated into body-fixed frames
    F13_r = F13tr*R1;
    F14_r = F14tr*R1;
    F23_r = F23tr*R1;
    F24_r = F24tr*R1;

    F31_r = -F13tr*R2;
    F41_r = -F14tr*R2;
    F32_r = -F23tr*R2;
    F42_r = -F24tr*R2;

    % Torques (z-component always zero for linear diatomic)
    M1(1) = -bondLength/2*(F13_r(2)+F14_r(2)) + bondLength/2*(F23_r(2)+F24_r(2));
    M1(2) =  bondLength/2*(F13_r(1)+F14_r(1)) - bondLength/2*(F23_r(1)+F24_r(1));
    M1(3) = 0;

    M2(1) = -bondLength/2*(F31_r(2)+F32_r(2)) + bondLength/2*(F41_r(2)+F42_r(2));
    M2(2) =  bondLength/2*(F31_r(1)+F32_r(1)) - bondLength/2*(F41_r(1)+F42_r(1));
    M2(3) = 0;
end
