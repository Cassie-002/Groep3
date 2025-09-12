%% Clear workspace, command window and figures
close all; clear; clc;
format long G

molecule = "N2";   % <---- verander dit naar "N2" of "O2" of iets anders kan ook

%% Parameter database
switch molecule
    case "H2"
        params.m_atom = 1.6738e-27;       % [kg]
        params.bondLength = 0.741e-10;    % [m]
        params.sigma = 3.06e-10;          % [m]params
        params.eps_K = 34.00;             % [K]
    case "N2"
        params.m_atom = 2.3250e-26;       
        params.bondLength = 1.10e-10;     
        params.sigma = 3.69e-10;          
        params.eps_K = 95.93;             
    case "O2"
        params.m_atom = 2.6567e-26;       
        params.bondLength = 1.21e-10;     
        params.sigma = 3.46e-10;          
        params.eps_K = 118;              
    otherwise
        error("Molecule not on the list yet")
end

%% parameters
params.m_mol = 2*params.m_atom;                      % molecular mass [kg]
params.mu    = (params.m_mol^2)/(2*params.m_mol);    % reduced mass [kg]
params.I     = 0.5*params.bondLength^2*params.m_atom;         % moment of inertia [kg m^2]

%% Constants
kB = 1.38064852e-23;   % Boltzmann constant [J/K]

%% # of collisions
ncoll = 500;

%% Database initialization (Nx11 Table)
varTypes = repmat("double",1,11);
varNames = ["b","Etr","EtrA","EtrB","Er1","Er2","Etrp","EtrAp","EtrBp","Er1p","Er2p"];
Table = table('Size',[ncoll,length(varNames)], ...
              'VariableTypes',varTypes,'VariableNames',varNames);

%% Simulation settings
dt = 0.1e-15;        % Time-step [s]
dt2 = dt*dt;         % Time-step squared [s^2]
tsim = 2e-12;        % Max. simulation time [s]
nSteps = tsim/dt;    % Max. number of steps

tic
for i = 1:ncoll
    %% Configuration properties
    Etr_K_max = 5950;                       % Maximum translational energy per molecule - 50 [K]
    Etr_K1 = 50+rand*Etr_K_max;             % Translational energy molecule 1 [K]
    Etr_J1 = Etr_K1*kB;                     % Energy molecule 1 [J]
    vtr1 = sqrt(2*Etr_J1/params.m_mol);     % Velocity molecule 1 [m/s]

    Etr_K2 = 50+rand*Etr_K_max;             % Translational energy molecule 2 [K]
    Etr_J2 = Etr_K2*kB;                     % Energy molecule 2 [J]
    vtr2 = sqrt(2*Etr_J2/params.m_mol);     % Velocity molecule 2 [m/s]

    bmax = 1.5*params.sigma;                % max. impact parameter [m]
    b = rand*bmax;                          % Actual impact parameter [m]
    bvec(i) = {b/params.sigma};

    Erot_K_max = 3000;                      % Maximum rotational energy [K]
    Erot_tot_1 = rand*Erot_K_max*kB;        % Rotational Energy molecule 1 [J]
    Erot_tot_2 = rand*Erot_K_max*kB;        % Rotational Energy molecule 2 [J]

    frac11 = rand; frac21 = rand;           % Energy fractions
    Er11 = frac11*Erot_tot_1;    Er12 = (1-frac11)*Erot_tot_1;
    Er21 = frac21*Erot_tot_2;    Er22 = (1-frac21)*Erot_tot_2;

    w11 = ((rand>0.5)*2-1)*sqrt(2*Er11/params.I);
    w12 = ((rand>0.5)*2-1)*sqrt(2*Er12/params.I);
    w21 = ((rand>0.5)*2-1)*sqrt(2*Er21/params.I);
    w22 = ((rand>0.5)*2-1)*sqrt(2*Er22/params.I);

    w1 = [w11;w12;0];       
    w2 = [w21;w22;0];       

    % Molecule masses
    m1 = params.m_mol;
    m2 = params.m_mol;

    %% Initial conditions
    X1 = [params.sigma*-2, 0, -b/2];
    X2 = [params.sigma*2,  0, b/2];

    X11_0 = [0, 0, 0.5*params.bondLength];
    X12_0 = [0, 0, -0.5*params.bondLength];
    X21_0 = [0, 0, 0.5*params.bondLength];
    X22_0 = [0, 0, -0.5*params.bondLength];

    % Random orientations
    R1 = getRandRotMat;     
    R2 = getRandRotMat;     

    Xv11 = R1*X11_0';   Xv12 = R1*X12_0';
    Xv21 = R2*X21_0';   Xv22 = R2*X22_0';

    X11 = X1+Xv11';  X12 = X1+Xv12';    
    X21 = X2+Xv21';  X22 = X2+Xv22';

    V1=[vtr1,0,0];     
    V2=[-vtr2,0,0];        

    % Preallocation
    Ekin1 = zeros(1,round(nSteps));
    Ekin2 = zeros(1,round(nSteps)); 
    Erot1 = zeros(1,round(nSteps));
    Erot2 = zeros(1,round(nSteps));
    Elj13 = zeros(1,round(nSteps));
    Elj14 = zeros(1,round(nSteps));
    Elj23 = zeros(1,round(nSteps));
    Elj24 = zeros(1,round(nSteps));
    dr = 0; step = 0;

    % Initial forces
F13tr = getFij(X11, X21, params.sigma, params.eps_K);  
F14tr = getFij(X11, X22, params.sigma, params.eps_K);           
F23tr = getFij(X12, X21, params.sigma, params.eps_K);       
F24tr = getFij(X12, X22, params.sigma, params.eps_K); 

F1 = (F13tr+F14tr+F23tr+F24tr);
F2 = -F1;

[M1, M2] = getM(F13tr, F14tr, F23tr, F24tr, R1, R2, params.bondLength);

    %% Simulation loop
    while dr <= 5*params.sigma
        step = step + 1;
        dr = norm(X1-X2);

        % Energies
        Ekin1(step) = 0.5*m1*norm(V1)^2;
        Ekin2(step) = 0.5*m2*norm(V2)^2;
        Erot1(step) = 0.5*params.I*(w1(1)^2+w1(2)^2);
        Erot2(step) = 0.5*params.I*(w2(1)^2+w2(2)^2);

        % LJ interactions
        dr13 = norm(X11-X21);   Elj13(step)  = LJ_energy(dr13, params.sigma, params.eps_K);
        dr14 = norm(X11-X22);   Elj14(step)  = LJ_energy(dr14, params.sigma, params.eps_K);
        dr23 = norm(X12-X21);   Elj23(step)  = LJ_energy(dr23, params.sigma, params.eps_K);
        dr24 = norm(X12-X22);   Elj24(step)  = LJ_energy(dr24, params.sigma, params.eps_K);

        % Forces (je hebt getFij en getM al in je project)
        F13tr = getFij(X11, X21, params.sigma, params.eps_K);  
        F14tr = getFij(X11, X22, params.sigma, params.eps_K);           
        F23tr = getFij(X12, X21, params.sigma, params.eps_K);       
        F24tr = getFij(X12, X22, params.sigma, params.eps_K);  

        F1 = (F13tr+F14tr+F23tr+F24tr);
        F2 = -F1;

       [M1, M2] = getM(F13tr, F14tr, F23tr, F24tr, R1, R2, params.bondLength);

        %% Velocity-Verlet
        V1_ = V1 + 0.5*dt*getVdot(F1,m1);
        V2_ = V2 + 0.5*dt*getVdot(F2,m2);

        X1 = X1 + dt*V1_;
        X2 = X2 + dt*V2_;

        R1_ = R1+0.5*dt*getRdot(w1, R1);
        R2_ = R2+0.5*dt*getRdot(w2, R2);

        w1_ = w1 + 0.5*dt*getWdot(M1, params.I)';
        w2_ = w2 + 0.5*dt*getWdot(M2, params.I)';

        R1 = R1 + dt*getRdot(w1_, R1_);
        R2 = R2 + dt*getRdot(w2_, R2_);

        Xv11 = R1*X11_0'; Xv12 = R1*X12_0';
        Xv21 = R2*X21_0'; Xv22 = R2*X22_0';

        X11 = X1+Xv11';  X12 = X1+Xv12';    
        X21 = X2+Xv21';  X22 = X2+Xv22';

        F13tr_ = getFij(X11, X21, params.sigma, params.eps_K); 
        F14tr_ = getFij(X11, X22, params.sigma, params.eps_K);      
        F23tr_ = getFij(X12, X21, params.sigma, params.eps_K);     
        F24tr_ = getFij(X12, X22, params.sigma, params.eps_K); 

        F1_ = (F13tr_+F14tr_+F23tr_+F24tr_);
        F2_ = -F1_;

        [M1_, M2_] = getM(F13tr_, F14tr_, F23tr_, F24tr_, R1, R2, params.bondLength);

        V1 = V1_ + 0.5*dt*getVdot(F1_,m1);
        V2 = V2_ + 0.5*dt*getVdot(F2_,m2);

        w1 = w1_ + 0.5*dt*getWdot(M1_, params.I)'; 
        w2 = w2_ + 0.5*dt*getWdot(M2_, params.I)'; 
    end

    %% Save results
    Ekin = Ekin1 + Ekin2;
    Erot = Erot1 + Erot2;
    Elj  = Elj13 + Elj14 + Elj23 + Elj24;
    Etot = Ekin  + Erot  + Elj;  

    Etrrvec(i)  = {Ekin(1)/kB};
    EtrAvec(i)  = {Ekin1(1)/kB};
    EtrBvec(i)  = {Ekin2(1)/kB};
    Etrrpvec(i) = {Ekin(step)/kB};
    EtrApvec(i) = {Ekin1(step)/kB};
    EtrBpvec(i) = {Ekin2(step)/kB};
    Er1vec(i)   = {Erot1(1)/kB};
    Er2vec(i)   = {Erot2(1)/kB};      
    Er1pvec(i)  = {Erot1(step)/kB};
    Er2pvec(i)  = {Erot2(step)/kB};
end
toc

%% Fill table
Table(:,1)  = bvec';
Table(:,2)  = Etrrvec';
Table(:,3)  = EtrAvec';
Table(:,4)  = EtrBvec';
Table(:,5)  = Er1vec';
Table(:,6)  = Er2vec';
Table(:,7)  = Etrrpvec';
Table(:,8)  = EtrApvec';
Table(:,9)  = EtrBpvec';
Table(:,10) = Er1pvec';
Table(:,11) = Er2pvec';

%% Save collision, verander deze elke keer!!
writetable(Table, 'collision_dataset.csv');

%% Inelastic collision fraction
E_xchanged = Table.Etr./Table.Etrp;
inelastic_count = sum((E_xchanged>0.99) & (E_xchanged<1.01));
inelastic_frac = inelastic_count/length(E_xchanged);

%% Visualization (same as je had)
figure(1)
set(gcf,'Position',[100 100 560*2.5 420*1])
t = tiledlayout(1,3);

nexttile; dscatter(Table.Etr,Table.Etrp)
box on; xlabel("E_{tr}/k_B (K)"); ylabel("E_{tr}'/k_B (K)")
set(gca,'FontSize',15,'linewidth',1)
title('Relative translational energy')

nexttile; dscatter(Table.EtrA,Table.EtrAp)
box on; xlabel("E_{tr,A}/k_B (K)"); ylabel("E_{tr,A}'/k_B (K)")
set(gca,'FontSize',15,'linewidth',1)
title('translational energy (A)')

nexttile; dscatter(Table.EtrB,Table.EtrBp)
box on; xlabel("E_{tr,B}/k_B (K)"); ylabel("E_{tr,B}'/k_B (K)")
set(gca,'FontSize',15,'linewidth',1)
title('translational energy (B)')
title(t,'Energy correlation graphs','fontsize',20,'fontweight','bold')

figure(2)
set(gcf,'Position',[100 100 560*2.5 420*1])
t = tiledlayout(1,3);
nexttile; dscatter(Table.b,(Table.Etrp-Table.Etr))
xlabel("Impact parameter b/\sigma_{LJ}"); ylabel("\Delta E_{tr}")
set(gca,'FontSize',15,'linewidth',1); box on
title('Relative translational energy')

nexttile; dscatter(Table.b,(Table.Er1p-Table.Er1))
xlabel("Impact parameter b/\sigma_{LJ}"); ylabel("\Delta E_{r,A}")
set(gca,'FontSize',15,'linewidth',1); box on
title('Rotational energy (A)')

nexttile; dscatter(Table.b,(Table.Er2p-Table.Er2))
xlabel("Impact parameter b/\sigma_{LJ}"); ylabel("\Delta E_{r,B}")
set(gca,'FontSize',15,'linewidth',1); box on
title('Rotational energy (B)')
title(t,'Change in energy vs impact parameter','fontsize',20,'fontweight','bold')
