%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  ADRC-based search algorithm (ASA) source codes version 0.1(beta)
%
%  Developed in:	MATLAB 9.13 (R2022b)
%
%  Programmer:		Jing Xiang
%
%  Original paper:	Jing Xiang,
%                   ADRC-based search algorithm: A novel metaheuristic
%                   algorithm based on active disturbance rejection control
%                   algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [TargetX, TargetF, ConvergenceCurve] = ASA(fun, nvars, lb, ub, N, T, varargin)
p = inputParser;
addParameter(p, 'Verbose', false);
addParameter(p, 'Verfig',     false);
addParameter(p, 'SEF_kp',       0.165);    % First-order channel equivalent feedback gain (LADRC linear feedback)
addParameter(p, 'ESO_Wo',       0.0775);     % Controller bandwidth ωc (the larger the response, the faster the noise/jitter ↑)
addParameter(p, 'gain',         0.012);      % Estimated control gain
addParameter(p, 'Err_td',       0.05);      % Gain coefficient of the first-order tracking differentiator (LTD)

parse(p, varargin{:});
opt = p.Results;

kp = opt.SEF_kp;
Wo = opt.ESO_Wo;
b = opt.gain;
r = opt.Err_td;

lb = reshape(lb,1,[]);
ub = reshape(ub,1,[]);
ConvergenceCurve = inf(1,T);

%% initialization
X = lb + rand(N,nvars).*(ub-lb);
f = arrayfun(@(i) fun(X(i,:)), 1:N)';
[TargetF,gid] = min(f);
TargetX = X(gid,:);

%% State initialization
z1 = zeros(N,1); 
z2 = zeros(N,1);         % LESO state: z1≈y, z2≈ total perturbation f
x1 = zeros(N,1);
u  = zeros(N,1);         % control input

beta1 = 2*Wo;           % The second-order LESO gain of first-order objects
beta2 = Wo^2;

%% Loop
for t=1:T
    % Update the global optimum
    [genBestF,gid] = min(f);
    if genBestF<TargetF
        TargetF=genBestF; TargetX=X(gid,:);
    end
    ConvergenceCurve(t)=TargetF;

    % 1) First-order LTD
    e1 = x1 - TargetX ;
    x1 = x1 - e1.*r;

    % 2) Second-order ESO Observation Update (Discrete Euler)
    e_y  = z1 - X;                               % output error
    z1   = z1 + ( z2 + b.*u - beta1.*e_y );      % z1[k+1]
    z2   = z2  - beta2.*e_y;                     % z2[k+1]

    % 3) LADRC control law (linear feedback + disturbance compensation) first-order LSEF
    u    = ( kp.*rand(N,1).*( x1 - z1 ) - z2.*rand(N,1) ) / b.*rand(N,1);

    % ---------- Lévy Flight ----------
    a = (log(T-t+2)/log(T))^2;
    out0 = (cos(1-t/T)+ a *rand(N,nvars).*levy_step(N,nvars,1.5)).*e1;
    % ---------- Hybrid update ----------
    a_t = rand(N,1)*cos(t/T);
    X = X + a_t.*u + (1-a_t).*out0;

    % Boundary treatment
    X = min(max(X,lb),ub);

    % Evaluate
    for i=1:N, f(i)=fun(X(i,:)); end

    if opt.Verbose && mod(t,round(T/10))==0
        fprintf('Iter %d/%d | Best=%.6g\n',t,T,TargetF);
    end
end
end

%% Lévy Flight
function L=levy_step(n,d,beta)
sigma=( gamma(1+beta)*sin(pi*beta/2) / ...
   (gamma((1+beta)/2)*beta*2^((beta-1)/2)) )^(1/beta);
u=randn(n,d)*sigma; v=randn(n,d);
L=u./(abs(v).^(1/beta));
end
