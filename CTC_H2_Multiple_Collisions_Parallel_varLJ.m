%% Clear workspace, command window and figues
close all; clear; clc;
format long G

%% Specifying the number of collisions
ncoll = 1000;

%% Database initialization (Nx7 Table)
varTypes = ["double","double","double","double","double","double","double","double","double"];
varNames = ["b","Etr","Er1","Er2","Etrp","Er1p","Er2p","sigma_LJ","eps_LJ"];
Table = table('Size',[ncoll,length(varNames)],'VariableTypes',varTypes,'VariableNames',varNames);

%% Constants
m_H  = 1.6738e-27;              % Hydrogen atom mass [kg]
m_H2 = m_H*2;                   % Hydrogen molecule mass [kg]
kB = 1.38064852e-23;            % Boltzmann constant [m^2*kg/s^2/K]
mu_H2 = m_H2*m_H2/(m_H2+m_H2);  % Reduced mass H2 [kg]
d_H2 = 0.741*1e-10;             % Interatomic distance for H2 molecule [m]
I = 0.5*d_H2^2*m_H;             % Hydrogen moment of inertia [kg m^2]

%% Simulation settings
dt = 0.1e-15;        % Time-step [s]
dt2 = dt*dt;         % Time-step squared [s^2]
tsim = 2e-12;        % Max. simulation time [s]
nSteps = tsim/dt;    % Max. number of steps

tic
parfor i = 1:ncoll
%% Configuration properties
    Etr_K_max = 5900;                    % Maximum translational energy [K]
    Etr_K = 100+rand*Etr_K_max;         % Actual translational energy [K]
    Etr_J = Etr_K*kB;                   % Energy [J]
    vtr = sqrt(Etr_J/m_H2);             % Velocity [m/s]

    sigma_LJmax = 6*1e-10             %max. sigma [m] 3.8
    sigma_LJmin = 3*1e-10               %min. sigma [m]
    sigma_LJ = rand*(sigma_LJmax-sigma_LJmin)+sigma_LJmin % actual sigma[m]
    sigma(i) = {sigma_LJ};

    epsmax = 300                        %max. well depth 140
    epsmin = 30                         %min. well depth
    eps_LJ = rand*(epsmax-epsmin)+epsmin   %well depth [K]
    eps(i) = {eps_LJ};

    bmax = 1.5*sigma_LJ;             % max. impact parameter [m]
    b = rand*bmax;                      % Actual impact parameter [m]
    bvec(i) = {b/sigma_LJ};

    Erot_K_max = 3000;                    % Maximum rotational energy [K]
    Erot_tot_1 = rand*Erot_K_max*kB;     % Actual rotational Energy of molecule 1 [J]
    Erot_tot_2 = rand*Erot_K_max*kB;     % Actual rotational Energy of molecule 2 [J] 

    frac11 = rand;                      % Fraction of rotational energy in mode 1 (molecule 1)
    frac21 = rand;                      % Fraction of rotational energy in mode 1 (molecule 1)          

    Er11 = frac11*Erot_tot_1;    Er12 = (1-frac11)*Erot_tot_1; % Rotational energies [J]
    Er21 = frac21*Erot_tot_2;    Er22 = (1-frac21)*Erot_tot_2; % Rotational energies [J]
    
    w11 = ((rand(1,1) > 0.5)*2 - 1)*sqrt(2*Er11/I);       % Angular velocity [rad/s]
    w12 = ((rand(1,1) > 0.5)*2 - 1)*sqrt(2*Er12/I);       % Angular velocity [rad/s]  
    w21 = ((rand(1,1) > 0.5)*2 - 1)*sqrt(2*Er21/I);       % Angular velocity [rad/s]
    w22 = ((rand(1,1) > 0.5)*2 - 1)*sqrt(2*Er22/I);       % Angular velocity [rad/s]  
  
    w1 = [w11;w12;0];       % Angular velocity vector
    w2 = [w21;w22;0];       % Angular velocity vector
    
    % Molecule 1 with 2 atoms 1 and 2
    m11 = m_H;               % Atom mass [kg]
    m12 = m_H;               % Atom mass [kg]
    m1  = m11+m12;                  % Molecular mass [kg]

    % Molecule 2 with 2 atoms 1 and 2
    m21 = m_H;               % Atom mass [kg]
    m22 = m_H;               % Atom mass [kg]
    m2  = m21+m22;                  % Molecular mass [kg]


    %% Initial conditions
    X1 = [sigma_LJ*-2, 0, -b/2];    % COM position molecule 1 [m]
    X2 = [sigma_LJ*2,  0, b/2];     % COM position molecule 2 [m]
  
    X11_0 = [0, 0, 0.5*d_H2];       % Atom position 1 [m]
    X12_0 = [0, 0, -0.5*d_H2];      % Atom position 2 [m]

    X21_0 = [0, 0, 0.5*d_H2];      % Atom position 3 [m]
    X22_0 = [0, 0, -0.5*d_H2];     % Atom position 4 [m]
    
    % Set random initial rotational orientation    
    R1 = getRandRotMat;     
    R2 = getRandRotMat;     

    Xv11 = R1*X11_0';
    Xv12 = R1*X12_0';
    Xv21 = R2*X21_0';
    Xv22 = R2*X22_0';

    X11 = X1+Xv11';
    X12 = X1+Xv12';    
    X21 = X2+Xv21';
    X22 = X2+Xv22';
    
    V1=[vtr,0,0];     % Initial velocity vector [m/s]
    V2=[-vtr,0,0];    % Initial velocity vector [m/s]
        
    % Preallocation data arrays
    Ekin1 = zeros(1,round(nSteps));
    Ekin2 = zeros(1,round(nSteps)); 
    Erot1 = zeros(1,round(nSteps));
    Erot2 = zeros(1,round(nSteps));
    Elj13 = zeros(1,round(nSteps));
    Elj14 = zeros(1,round(nSteps));
    Elj23 = zeros(1,round(nSteps));
    Elj24 = zeros(1,round(nSteps));
    dr12v = zeros(1,round(nSteps));
    dr13v = zeros(1,round(nSteps));
    dr14v = zeros(1,round(nSteps));
    dr23v = zeros(1,round(nSteps));
    dr24v = zeros(1,round(nSteps));
    dr34v = zeros(1,round(nSteps));
    drABv = zeros(1,round(nSteps));


    dr = 0;
    step = 0;
    %% Simulation
    while dr <= 5*sigma_LJ
        step = step + 1;
        dr = norm(X1-X2);
        drABv(step) = dr;
        % Extracting values at timestep (t)
        Ekin1(step) = 0.5*m1*norm(V1)^2;            % Translational kinetic energy [J]
        Ekin2(step) = 0.5*m2*norm(V2)^2;            % Translational kinetic energy [J]
        Erot1(step) = 0.5*I*(w1(1)^2+w1(2)^2);      % Rotational kinetic energy [J]
        Erot2(step) = 0.5*I*(w2(1)^2+w2(2)^2);      % Rotational kinetic energy [J]

        % Computing interatomic distances and energies at timestep (t)
        dr13 = norm(X11-X21);   dr13v(step)=dr13;     Elj13(step)  = varLJ_e(dr13,sigma_LJ,eps_LJ); 
        dr14 = norm(X11-X22);   dr14v(step)=dr14;     Elj14(step)  = varLJ_e(dr14,sigma_LJ,eps_LJ);
        dr23 = norm(X12-X21);   dr23v(step)=dr23;     Elj23(step)  = varLJ_e(dr23,sigma_LJ,eps_LJ);
        dr24 = norm(X12-X22);   dr24v(step)=dr24;     Elj24(step)  = varLJ_e(dr24,sigma_LJ,eps_LJ);
        
        % Computing interatomic distance for molecules 1 and 2
        dr12v(step) = norm(X11-X12);
        dr34v(step) = norm(X21-X22);

        % Computing the interatomic forces in the inertial frame  at 
        % time-step (t) using the modeled interatomic potential. 
        % For translational motion, the sum of the forces determines the 
        % acceleration of the molecules' COM.

        F13tr = getVarFij(X11, X21,sigma_LJ,eps_LJ);  
        F14tr = getVarFij(X11, X22,sigma_LJ,eps_LJ);           
        F23tr = getVarFij(X12, X21,sigma_LJ,eps_LJ);       
        F24tr = getVarFij(X12, X22,sigma_LJ,eps_LJ); 
           
        F1 = (F13tr+F14tr+F23tr+F24tr);
        F2 = -F1;
      
        
        % Computing the momenta in the body-fixed frames using the forces
        % in inertial frame and the orientations of the molecules. Momenta
        % are computed at time-step (t)

        [M1, M2] = getM(F13tr, F14tr, F23tr, F24tr, R1, R2, d_H2);

        %% Velocity-verlet algorithm
        % The half-step velocities at (t+0.5dt) are computed using
        % the acceleration computed with the force at time-step (t). This
        % step may be eliminated by substituting these lines further in the
        % code.
        V1_ = V1 + 0.5*dt*getVdot(F1,m1);
        V2_ = V2 + 0.5*dt*getVdot(F2,m2);
        
        % The positions at (t+dt) are updated using the previous positions
        % at (t) and the half-step velocities at (t+0.5dt).
        X1 = X1 + dt*V1_;
        X2 = X2 + dt*V2_;

        % To compute the half-step derivative of the rotation matrix R at
        % (t+0.5dt), an estimation of the rotation matrix at (t+0.5dt) is
        % required. This computation is identical to that of the half-step 
        % velocities.
        R1_ = R1+0.5*dt*getRdot(w1, R1);
        R2_ = R2+0.5*dt*getRdot(w2, R2);
        
        % The half-step angular velocities at (t+0.5dt) are computed
        % identical to the translational velocities.
        w1_ = w1 + 0.5*dt*getWdot(M1, I)';
        w2_ = w2 + 0.5*dt*getWdot(M2, I)';
        
        % The rotational orientations at (t+dt) are updated identical to
        % the translational positions.
        R1 = R1 + dt*getRdot(w1_, R1_);
        R2 = R2 + dt*getRdot(w2_, R2_);
        
        % Atomic positions at (t+dt) are updated using the positions of
        % the centers of masses at (t+dt), and the rotation matrices at
        % (t+dt).
        Xv11 = R1*X11_0';
        Xv12 = R1*X12_0';
        Xv21 = R2*X21_0';
        Xv22 = R2*X22_0';
        
        X11 = X1+Xv11';
        X12 = X1+Xv12';    
        X21 = X2+Xv21';
        X22 = X2+Xv22';
        
        % Forces and momenta at (t+dt) are computed with the updated atomic
        % positions at (t+dt) and the interatomic potential.
        F13tr_ = getVarFij(X11, X21,sigma_LJ,eps_LJ); 
        F14tr_ = getVarFij(X11, X22,sigma_LJ,eps_LJ);      
        F23tr_ = getVarFij(X12, X21,sigma_LJ,eps_LJ);     
        F24tr_ = getVarFij(X12, X22,sigma_LJ,eps_LJ); 
        
        F1_ = (F13tr_+F14tr_+F23tr_+F24tr_);
        F2_ = -F1_;
        
        [M1_, M2_] = getM(F13tr_, F14tr_, F23tr_, F24tr_, R1, R2, d_H2);

        % Translational and angular velocities at (t+dt) 
        % are updated using the new forces and momenta at (t+dt).
        V1 = V1_ + 0.5*dt*getVdot(F1_,m1);
        V2 = V2_ + 0.5*dt*getVdot(F2_,m2);
        
        w1 = w1_ + 0.5*dt*getWdot(M1_, I)'; 
        w2 = w2_ + 0.5*dt*getWdot(M2_, I)'; 
    end   
    
    % Assigning data to dataset and preallocated data arrays
    Ekin = Ekin1 + Ekin2;
    Erot = Erot1 + Erot2;
    Elj  = Elj13 + Elj14 + Elj23 + Elj24;
    Etot = Ekin  + Erot  + Elj;  
    
    Etrrvec(i) = {Ekin(1)/kB};
    Etrrpvec(i) = {Ekin(step)/kB};
    Er1vec(i) = {Erot1(1)/kB};
    Er2vec(i) = {Erot2(1)/kB};      
    Er1pvec(i) = {Erot1(step)/kB};
    Er2pvec(i) = {Erot2(step)/kB};
    
end
toc

% Fill table
Table(:,1) = bvec';
Table(:,2) = Etrrvec';
Table(:,3) = Er1vec';
Table(:,4) = Er2vec';
Table(:,5) = Etrrpvec';
Table(:,6) = Er1pvec';
Table(:,7) = Er2pvec';
Table(:,8) = sigma';
Table(:,9) = eps';

% Save collision dataset
string = ['collision_dataset_',num2str(ncoll)];
writetable(Table, string);

%% Visualization
% figure(1)
% set(gcf,'Position',[100 100 560*2.5 420*1])
% t = tiledlayout(1,3);
% 
% nexttile;
% dscatter(Table.Etr,Table.Etrp)
% box on;
% xlabel("E_{tr}/k_B (K)")
% ylabel("E_{tr}'/k_B (K)")
% set(gca, 'FontSize',15);
% set(gca,'linewidth',1)
% title('Relative translational energy')
% 
% nexttile;
% dscatter(Table.Er1,Table.Er1p)
% box on;
% xlabel("E_{r,A}/k_B (K)")
% ylabel("E_{r,A}'/k_B (K)")
% set(gca, 'FontSize',15);
% set(gca,'linewidth',1)
% title('Rotational energy (A)')
% 
% nexttile;
% dscatter(Table.Er2,Table.Er2p)
% box on;
% xlabel("E_{r,B}/k_B (K)")
% ylabel("E_{r,B}'/k_B (K)")
% set(gca, 'FontSize',15);
% set(gca,'linewidth',1)
% title('Rotational energy (B)')
% title(t,'Energy correlation graphs','fontsize',20,'fontweight','bold')
% 
% 
% figure(2)
% set(gcf,'Position',[100 100 560*2.5 420*1])
% t = tiledlayout(1,3);
% nexttile;
% dscatter(Table.b,(Table.Etrp-Table.Etr))
% xlabel("Impact parameter b/\sigma_{LJ}")
% ylabel("\Delta E_{tr} = E_{tr}'-E_{tr}")
% set(gca, 'FontSize',15);
% box on
% set(gca,'linewidth',1)
% title('Relative translational energy')
% 
% nexttile;
% dscatter(Table.b,(Table.Er1p-Table.Er1))
% xlabel("Impact parameter b/\sigma_{LJ}")
% ylabel("\Delta E_{r,A} = E_{r,A}'-E_{r,A}")
% set(gca, 'FontSize',15);
% box on
% set(gca,'linewidth',1)
% title('Rotational energy (A)')
% 
% nexttile;
% dscatter(Table.b,(Table.Er2p-Table.Er2))
% xlabel("Impact parameter b/\sigma_{LJ}")
% ylabel("\Delta E_{r,B} = E_{r,B}'-E_{r,B}")
% set(gca, 'FontSize',15);
% box on
% set(gca,'linewidth',1)
% title('Rotational energy (B)')
% title(t,'Change in energy as a function of impact parameter','fontsize',20,'fontweight','bold')

figure(3)
set(gcf,'Position',[100 100 560*2.5 420*1])
t = tiledlayout(1,3);
nexttile;
dscatter(Table.sigma_LJ,(Table.Etrp-Table.Etr))
xlabel("sigma_{LJ}")
ylabel("\Delta E_{tr} = E_{tr}'-E_{tr}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Relative translational energy')

nexttile;
dscatter(Table.sigma_LJ,(Table.Er1p-Table.Er1))
xlabel("sigma_{LJ}")
ylabel("\Delta E_{r,A} = E_{r,A}'-E_{r,A}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Rotational energy (A)')

nexttile;
dscatter(Table.sigma_LJ,(Table.Er2p-Table.Er2))
xlabel("sigma_{LJ}")
ylabel("\Delta E_{r,B} = E_{r,B}'-E_{r,B}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Rotational energy (B)')
title(t,'Change in energy as a function of sigma_{LJ}','fontsize',20,'fontweight','bold')

figure(4)
set(gcf,'Position',[100 100 560*2.5 420*1])
t = tiledlayout(1,3);
nexttile;
dscatter(Table.eps_LJ,(Table.Etrp-Table.Etr))
xlabel("epsilon_{LJ}")
ylabel("\Delta E_{tr} = E_{tr}'-E_{tr}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Relative translational energy')

nexttile;
dscatter(Table.eps_LJ,(Table.Er1p-Table.Er1))
xlabel("epsilon_{LJ}")
ylabel("\Delta E_{r,A} = E_{r,A}'-E_{r,A}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Rotational energy (A)')

nexttile;
dscatter(Table.eps_LJ,(Table.Er2p-Table.Er2))
xlabel("epsilon_{LJ}")
ylabel("\Delta E_{r,B} = E_{r,B}'-E_{r,B}")
set(gca, 'FontSize',15);
box on
set(gca,'linewidth',1)
title('Rotational energy (B)')
title(t,'Change in energy as a function of well depth epsilon_{LJ}','fontsize',20,'fontweight','bold')