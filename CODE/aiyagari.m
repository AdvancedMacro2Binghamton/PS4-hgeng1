% PROGRAM NAME: aiyagari.m
clear all;
close all;
clc;

% PARAMETERS
beta = .99; %discount factor 
alpha=1/3; %production elasticity
sigma = 2; % coefficient of risk aversion
delta=.025; %depreciation rate
rho_epsi=0.5; %AR(1) autoregressive
sigma_epsi=0.2; %AR(1) std. devs of residual

%exogenous variable grid
num_z=5;
[z_grid, PI]=TAUCHEN(num_z,rho_epsi,sigma_epsi,3);
z_grid=exp(z_grid');
% invariant distribution is 
PI_inv=PI^1000;
PI_inv=PI_inv(1,:)';
N_s=z_grid*PI_inv; % aggregate labor supply (Q3.)
% ASSET VECTOR
a_lo = 0; %lower bound of grid points
a_hi = 80;%(guess) upper bound of grid points
num_a = 500;

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

%INITIAL GUESS OF K AND FACTOR PRICES
K_min=20;
K_max=50;
K_tol=1;
while abs(K_tol)>.01
    if K_max-K_min<0.00001
       break
   end
    % K_guess and correspoding factor prices
    K_guess=(K_min+K_max)/2;
    interest= alpha*K_guess^(alpha-1)*N_s^(1-alpha)+(1-delta);
    wage=(1-alpha)*K_guess^alpha*N_s^(-alpha);
    
    % CURRENT RETURN (UTILITY) FUNCTION
    cons = bsxfun(@minus, interest* a', a);
    cons = bsxfun(@plus, cons, permute(z_grid, [1 3 2])*wage);
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0)=-Inf;
    % INITIAL VALUE FUNCTION GUESS
    v_guess = zeros(num_z, num_a);
    
    % VALUE FUNCTION ITERATION
    v_tol = 1;
    while v_tol >.000001;
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE
        value_mat=ret+beta*repmat(permute((PI*v_guess),[3 2 1]), [num_a 1 1]);
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
       [vfn, pol_indx] = max(value_mat, [], 2); %max for each row
       vfn=permute(vfn, [3 1 2]);
       v_tol = max(abs(vfn-v_guess));
       v_tol = max(v_tol(:));
       
       v_guess = vfn;
    end;
    
    % KEEP DECSISION RULE
    pol_indx=permute(pol_indx, [3 1 2]);
    pol_fn = a(pol_indx);
    
    % SET UP INITITAL DISTRIBUTION
    MU=zeros(num_z,num_a);
    MU(:)=1/(num_z*num_a);
    
    dis=1;
  while dis>0.0000001 
      % ITERATE OVER DISTRIBUTIONS
      MuNew = zeros(size(MU));
     [z_ind, a_ind, mass] = find(MU); % find non-zero indices
    
    for ii = 1:length(z_ind)
        apr_ind = pol_indx(z_ind(ii),a_ind(ii)); % which a prime does the policy fn prescribe?
        
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
            (PI(z_ind(ii), :)*mass(ii))';
        
    end
    dis = max(max(abs(MU-MuNew)));
    MU=MuNew;
  end
   %Market clears
   K=sum(sum(MU.*pol_fn));
   K_tol=K-K_guess;
   if K_tol>0;
       K_min=K_guess;
   else K_max=K_guess;
   end
end

%Q6 (1) policy function
figure(1)
plot(a,pol_fn)
z_name=cellstr(num2str(z_grid'));
legend(z_name,'location','southeast')
title(['Policy Function for Productivity z'])

%Q6. (2) gini coefficient and lorenz curve
pop=reshape(MU',[num_z*num_a,1]);
wealth=reshape(repmat(a,num_z,1)',[num_z*num_a,1]);

%%%%%%% Distribution Plot%%%%%
mu=sum(MU);
figure(2)
bar(a,mu)
title('Distribution of Assets')
%%%%%%% Gini coefficient and lorenz curve%%%%
WEALTH=sortrows([wealth,pop,pop.*wealth]);
WEALTH=cumsum(WEALTH);
pw=WEALTH(:,2);
pw=pw(end);
WEALTH(:,2)=WEALTH(:,2)/pw;
w=WEALTH(:,3);
w=w(end);
WEALTH(:,3)=WEALTH(:,3)/w;
gini_wealth2 = 1 - sum((WEALTH(1:end-1,3)+WEALTH(2:end,3)) .* diff(WEALTH(:,2)));

figure(3)
suptitle('Lorenz Curve' )
area(WEALTH(:,2),WEALTH(:,3),'FaceColor',[0.5,0.5,1.0])
hold on
plot([0,1],[0,1],'--k')
axis square
title(['Wealth, Gini=',num2str(gini_wealth2)])
hold off

y=z_grid*wage
Y=repmat(y',[1,num_a]);
A=repmat(a,[num_z,1])
c=Y+interest*A-pol_fn;
cf=c(:,pol_indx');
cf1=reshape(cf,[num_z num_a num_z]);
i=1;
while i < num_z+1
c1(i,:)=PI(i,:)*cf1(:,:,i);
i=i+1;
end
Eulererror=sum(sum(abs(c.^(-sigma)-beta*c1.^(-sigma)*interest).*MU))
