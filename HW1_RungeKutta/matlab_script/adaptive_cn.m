clear all; close all; clc;

%% 1. Problem configuration (Ex 1 equation: y' = -10 * y^2)
f = @(t, u) -10 * u^2;

% Jacobian (Derivative with respect to u): d(-10u^2)/du = -20u
Jf = @(u) -20 * u;

u0 = 1;             % Initial condition scalar
t0 = 0; 
T = 100;
h_start = 1e-3;
TOL = 1e-8;         % Target tolerance for adaptive method

%% 2. Generation of "Exact Solution" 
get_exact = @(t) 1 ./ (10 .* t + 1);

%% 3. Constant Step Crank-Nicolson Solution
fprintf('Starting Constant Step CN (h=%e)...\n', h_start);
t_fix = t0:h_start:T;
N_fix = length(t_fix);

% CORRETTO: Inizializzazione scalare (1 riga) invece di 2
u_fix = zeros(1, N_fix); 
u_fix(1) = u0;

tic;
for n = 1:N_fix-1
    u_fix(n+1) = step_CN_Newton(t_fix(n), u_fix(n), h_start, f, Jf);
end
time_fix = toc;

% Calculate Constant Step Error
err_fix = zeros(1, N_fix);
for i = 1:N_fix
    u_true = get_exact(t_fix(i));
    err_fix(i) = abs(u_fix(i) - u_true); % Norm non necessaria per scalari
end
fprintf('Constant Step CN completed. Steps: %d. CPU Time: %.4f s\n', N_fix, time_fix);

%% 4. Adaptive Step Crank-Nicolson Solution
fprintf('Starting Adaptive CN (TOL=%e)...\n', TOL);
t_ada = [t0];
u_ada = [u0];
h_vec = []; % History of used stepsizes
h = h_start;
curr_t = t0;
curr_u = u0;
p = 2; % Order of Crank-Nicolson

tic;
while curr_t < T
    % Make sure not to exceed T in the last step
    if curr_t + h > T
        h = T - curr_t;
    end
    
    % --- Richardson Extrapolation ---
    % 1. Coarse Step (1 step of size h)
    y1 = step_CN_Newton(curr_t, curr_u, h, f, Jf);
    
    % 2. Fine Step (2 steps of size h/2)
    y_mid = step_CN_Newton(curr_t, curr_u, h/2, f, Jf);
    y2    = step_CN_Newton(curr_t + h/2, y_mid, h/2, f, Jf);
    
    % 3. Error Estimation
    % Formula: E = (y2 - y1) / (2^p - 1)
    error_est = (y2 - y1) / (2^p - 1);
    error_val = abs(error_est);
    
    % 4. Tolerance Check
    if error_val <= TOL
        % ACCEPT THE STEP
        curr_t = curr_t + h;
        % Solution Improvement (Extrapolation)
        curr_u = y2 + error_est; 
        
        % Save data
        t_ada = [t_ada, curr_t];
        u_ada = [u_ada, curr_u];
        h_vec = [h_vec, h];
        
        % Calculate new step (larger)
        % Safety factor 0.9
        if error_val == 0
             h = h * 2; % Se errore è 0 (macchina), raddoppia brutalmente
        else
             h = h * 0.9 * (TOL / error_val)^(1/(p+1));
        end
        
        % Limit maximum step growth to avoid jumping to infinity
        h = min(h, 10); 
    else
        % REJECT THE STEP
        % Recalculate smaller h
        if error_val == 0
            error_val = eps; % Evita divisione per zero
        end
        h = h * 0.9 * (TOL / error_val)^(1/(p+1));
    end
end
time_ada = toc;

% Calculate Adaptive Step Error
N_ada = length(t_ada);
err_ada = zeros(1, N_ada);
for i = 1:N_ada
    u_true = get_exact(t_ada(i));
    err_ada(i) = abs(u_ada(i) - u_true);
end
fprintf('CN Adaptive completed. Steps: %d. CPU Time: %.4f s\n', N_ada, time_ada);

%% 5. Plotting Results

% FIGURE 1: Error Comparison
figure(1);
semilogy(t_fix, err_fix, 'r-', 'LineWidth', 1); hold on;
semilogy(t_ada, err_ada, 'b.-', 'LineWidth', 0.5, 'MarkerSize', 6);
yline(TOL, 'w-', 'Tolerance');
title('Global Error vs Time');
xlabel('Time'); ylabel('Error (Log Scale)');
legend('Constant Step (h=1e-3)', 'Adaptive Step', 'Target TOL');
grid on;

% FIGURE 2: Stepsize Evolution
figure(2);
semilogy(t_ada(2:end), h_vec, 'c-', 'LineWidth', 1.5); % Semilogy consigliato qui
title('Adaptive Stepsize Evolution');
xlabel('Time'); ylabel('Step Size h (Log Scale)');
grid on;

% FIGURE 3: Efficiency Table
fprintf('\n--- EFFICIENCY COMPARISON ---\n');
fprintf('Method      | Steps | Max Error    | CPU Time\n');
fprintf('Constant    | %5d | %e | %.4f s\n', N_fix, max(err_fix), time_fix);
fprintf('Adaptive    | %5d | %e | %.4f s\n', N_ada, max(err_ada), time_ada);


%% Auxiliary Function: Single CN Step with Newton
function u_new = step_CN_Newton(t_now, u_old, h, f, Jf)
    % Solves u_new = u_old + h/2 * (f(u_old) + f(u_new))
    % using Newton
    
    max_iter = 10;
    tol_newton = 1e-10;
    
    u_new = u_old; % Initial guess
    I = 1; % Identity for scalar is 1
    
    for k = 1:max_iter
        % Residual F(u_new) = 0
        % f(t_now, u_old) -> u_old^2 is scalar compatible
        val_f_old = f(t_now, u_old);
        val_f_new = f(t_now+h, u_new);
        
        F = u_new - u_old - (h/2)*(val_f_old + val_f_new);
        
        % Jacobian J_F = I - h/2 * J_f(u_new)
        J = I - (h/2)*Jf(u_new);
        
        delta = - F / J; % Division for scalar
        u_new = u_new + delta;
        
        if abs(delta) < tol_newton
            return;
        end
    end
end