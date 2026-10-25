clear all; close all; clc;

%% 1. Definition of the Lotka-Volterra parameters
alpha = 0.2;
beta  = 0.01;
gamma = 0.004;
delta = 0.07;

% Initial Conditions and Time
u0 = [19; 22]; % [x; y]
t0 = 0;
T  = 300;
h  = 1e-3;     % Stepsize

N = ceil((T - t0) / h);
t = t0:h:T;

% Definition of the function f(t, u) as a function handle
% u(1) is x, u(2) is y
f = @(t, u) [ u(1) * (alpha - beta * u(2)); ...
              u(2) * (gamma * u(1) - delta) ];

% Definition of the Jacobian J_f(u) for Newton's method (Point c)
Jf = @(u) [ alpha - beta * u(2),   -beta * u(1); ...
            gamma * u(2),          gamma * u(1) - delta ];

%% 2. POINT (A): Runge-Kutta 4 (RK4)
% Initialization of solution matrix (2 rows, N+1 columns)
U_rk4 = zeros(2, N+1);
U_rk4(:, 1) = u0;

fprintf('Starting RK4 simulation...\n');
tic;
for n = 1:N
    tn = t(n);
    un = U_rk4(:, n);
    
    k1 = f(tn, un);
    k2 = f(tn + h/2, un + h/2 * k1);
    k3 = f(tn + h/2, un + h/2 * k2);
    k4 = f(tn + h,   un + h * k3);
    
    U_rk4(:, n+1) = un + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end
elapsed_rk4 = toc;
fprintf('RK4 completed in %.4f seconds.\n', elapsed_rk4);

%% 3. POINT (B): RK4 Plots
figure(1);
plot(t, U_rk4(1, :), 'b-', 'LineWidth', 1.5); hold on;
plot(t, U_rk4(2, :), 'r--', 'LineWidth', 1.5);
title('Lotka-Volterra Dynamics (RK4 Method)');
xlabel('Time t');
ylabel('Population');
legend('Prey (x)', 'Predators (y)');
grid on;

% Optional plot in the phase space (Limit cycle)
figure(2);
plot(U_rk4(1, :), U_rk4(2, :), 'c-');
title('Phase Space (RK4)');
xlabel('Prey x');
ylabel('Predators y');
grid on;

%% 4. POINT (C): Crank-Nicolson with Newton
% The method is: u_new = u_old + (h/2) * [f(u_old) + f(u_new)]
% We need to find the zero of the Residual function F(u_new) = 0
% F(u_new) = u_new - u_old - (h/2)*[f(u_old) + f(u_new)]

U_cn = zeros(2, N+1);
U_cn(:, 1) = u0;

% Parametri per Newton
max_iter = 10;
tol = 1e-8;

fprintf('\nStarting Crank-Nicolson simulation (Newton)...\n');
tic;
for n = 1:N
    u_old = U_cn(:, n);
    t_new = t(n+1);
    
    % Initial guess for Newton: use the previous solution (or FE)
    u_new = u_old; 
    
    % Newton loop
    for k = 1:max_iter
        % Compute the Residual F
        Fun_val = u_new - u_old - (h/2) * (f(t(n), u_old) + f(t_new, u_new));
        
        % Compute the Jacobian of the Residual: J_F = I - (h/2)*J_f
        Jacobian_F = eye(2) - (h/2) * Jf(u_new);
        
        % Solve the linear system J*delta = -F
        delta_u = - Jacobian_F \ Fun_val;
        
        % Update solution
        u_new = u_new + delta_u;
        
        % Check convergence
        if norm(delta_u, inf) < tol
            break;
        end
    end
    
    U_cn(:, n+1) = u_new;
end
elapsed_cn = toc;
fprintf('Crank-Nicolson completed in %.4f seconds.\n', elapsed_cn);

% Graphical comparison (first for T, second zoomed in for checking stability)
figure(3);
set(gcf, 'Position', [100, 100, 1200, 500]); 

% T (0-300)
subplot(1, 2, 1);
plot(t, U_rk4(1, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, U_cn(1, :), 'g--', 'LineWidth', 2);
title('Long-term Stability (T=0 to 300)');
xlabel('Time t');
ylabel('Prey Population (x)');
xlim([0, T]); 
legend('RK4', 'Crank-Nicolson', 'Location', 'best');
grid on;

% T (0-50)
subplot(1, 2, 2);
plot(t, U_rk4(1, :), 'b', 'LineWidth', 1.5); hold on;
plot(t, U_cn(1, :), 'g--', 'LineWidth', 2);
title('Detail/Zoom (T=0 to 50)');
xlabel('Time t');
ylabel('Prey Population (x)');
xlim([0, 50]); % Qui facciamo lo zoom
grid on;