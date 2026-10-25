%% Solution for the linear system of ODEs exercise

clear; clc; close all;

%% --- 0. Problem Setup ---

% Define problem parameters
nx = 60;
T_final = 0.1;

% Generate Matrix A using the provided commands
G = numgrid('S', nx);
A = delsq(G) * (nx-1)^2;

% Get system size
N = size(A, 1); 
y0 = ones(N, 1);

% Compute exact solution at T=0.1 using expm
fprintf('Computing the exact solution with expm.\n');
tic;
y_exact = expm(-T_final * A) * y0;
cpu_expm = toc;
fprintf('Exact solution computed in %.4f s.\n\n', cpu_expm);

%% --- Part (a): 4th Order Runge-Kutta (via ode45) ---

fprintf('--- Part (a): Solving with ode45 ---\n');
% Compute the eigenvalue of A with the largest modulus for stability analysis
lambda_max = eigs(A, 1, 'lm');

% The stability region for RK4 is h*lambda <= 2.78 for eigenvalues on the negative real axis.
% For our system y''=-Ay, the eigenvalues are -lambda_i(A).
h_max_rk4 = 2.78 / lambda_max;
fprintf('Largest eigenvalue magnitude of A: %.4e\n', lambda_max);
fprintf('Stability limit for explicit RK4: h <= %.4e\n\n', h_max_rk4);

% Solve with ode45 (Matlab's adaptive RK4/5 solver)
ode_func = @(t, y) -A*y;

tic;
[~, Y_ode45] = ode45(ode_func, [0, T_final], y0);
cpu_time_ode45 = toc;
y_final_ode45 = Y_ode45(end, :)';
error_ode45 = norm(y_final_ode45 - y_exact, inf);

fprintf('ode45 final error: %.4e\n', error_ode45);
fprintf('ode45 CPU time: %.4f s\n\n', cpu_time_ode45);

%% --- Part (b) & (c): Crank-Nicolson and BDF3 ---

k_values = 5:12;
h_values = 2.^(-k_values);
h_values = h_values(:); % Ensure h_values is a column vector for consistency
num_h = length(h_values);

% Preallocate results arrays: [error, time]
results = table('Size', [num_h, 8], ...
    'VariableTypes', repmat({'double'}, 1, 8), ...
    'VariableNames', {'CN_Chol_Err', 'CN_Chol_Time', 'CN_CG_Err', 'CN_CG_Time', ...
                      'BDF3_Chol_Err', 'BDF3_Chol_Time', 'BDF3_CG_Err', 'BDF3_CG_Time'});

% Main loop over different step sizes
for i = 1:num_h
    h = h_values(i);
    n_steps = round(T_final / h);
    fprintf('--- Processing h = %.2e (%d steps) ---\n', h, n_steps);

    % --- (b) Crank-Nicolson Method ---
    M_cn = speye(N) + (h/2) * A;
    RHS_cn_matrix = speye(N) - (h/2) * A;
    cg_tol_cn = h^3; % Tolerance proportional to LTE of CN

    % CN with Cholesky factorization
    tic;
    R_cn = chol(M_cn);
    y_cn = y0;
    for j = 1:n_steps
        b = RHS_cn_matrix * y_cn;
        y_cn = R_cn \ (R_cn' \ b);
    end
    results.CN_Chol_Time(i) = toc;
    results.CN_Chol_Err(i) = norm(y_cn - y_exact, inf);

    % CN with Conjugate Gradient
    tic;
    y_cn_cg = y0;
    for j = 1:n_steps
        b = RHS_cn_matrix * y_cn_cg;
        [y_cn_cg, ~] = pcg(M_cn, b, cg_tol_cn, 100);
    end
    results.CN_CG_Time(i) = toc;
    results.CN_CG_Err(i) = norm(y_cn_cg - y_exact, inf);

    % --- (c) BDF3 Method ---
    M_bdf3 = (11/6)*speye(N) + h * A;
    cg_tol_bdf3 = h^4; % Tolerance proportional to LTE of BDF3

    I_N = speye(N);
    M_start = I_N + (h/2)*A;
    RHS_op  = I_N - (h/2)*A;

    % Compute step 1 and step 2 using direct solve (\) for maximum precision
    % This ensures both BDF3 loops start from the exact same stable trajectory
    y1 = M_start \ (RHS_op * y0);
    y2 = M_start \ (RHS_op * y1);

    % --- Path 1: BDF3 with Cholesky factorization ---
    tic;
    R_bdf3 = chol(M_bdf3);
    y_hist = [y0, y1, y2]; % Initialize history
    for j = 3:n_steps
        b = 3*y_hist(:,3) - (3/2)*y_hist(:,2) + (1/3)*y_hist(:,1);
        y_next = R_bdf3 \ (R_bdf3' \ b);
        y_hist = [y_hist(:,2:3), y_next];
    end
    y_bdf3 = y_hist(:,3);
    results.BDF3_Chol_Time(i) = toc;
    results.BDF3_Chol_Err(i) = norm(y_bdf3 - y_exact, inf);

    % --- Path 2: BDF3 with Conjugate Gradient ---
    tic;
    y_hist_cg = [y0, y1, y2]; % Initialize history (SAME values as above)
    for j = 3:n_steps
        b = 3*y_hist_cg(:,3) - (3/2)*y_hist_cg(:,2) + (1/3)*y_hist_cg(:,1);
        [y_next, ~] = pcg(M_bdf3, b, cg_tol_bdf3, 100);
        y_hist_cg = [y_hist_cg(:,2:3), y_next];
    end
    y_bdf3_cg = y_hist_cg(:,3);
    results.BDF3_CG_Time(i) = toc;
    results.BDF3_CG_Err(i) = norm(y_bdf3_cg - y_exact, inf);
end

%% --- 4. Results and Analysis ---

% Display final table
fprintf('\n--- Summary Table ---\n');
results_with_h = [table(h_values, 'VariableNames', {'h'}), results];
disp(results_with_h);

% Plotting
figure;
loglog(h_values, results.CN_Chol_Err, '-o', 'DisplayName', 'CN-Cholesky', 'LineWidth', 1.5);
hold on;
loglog(h_values, results.CN_CG_Err, '--x', 'DisplayName', 'CN-CG', 'LineWidth', 1.5);
loglog(h_values, results.BDF3_Chol_Err, '-s', 'DisplayName', 'BDF3-Cholesky', 'LineWidth', 1.5);
loglog(h_values, results.BDF3_CG_Err, '--d', 'DisplayName', 'BDF3-CG', 'LineWidth', 1.5);

% Add reference lines for O(h^2) and O(h^3)
h_ref = [h_values(end), h_values(1)];
cn_ref = results.CN_Chol_Err(end) * (h_ref / h_values(end)).^2;
bdf3_ref = results.BDF3_Chol_Err(end) * (h_ref / h_values(end)).^3;
plot(h_ref, cn_ref, 'w--', 'DisplayName', 'O(h^2)');
plot(h_ref, bdf3_ref, 'w:', 'DisplayName', 'O(h^3)');

xlabel('Stepsize (h)');
ylabel('Final Error (inf-norm)');
title('Error vs. Stepsize for Implicit Methods');
legend('show', 'Location', 'southeast');
grid on;
set(gca, 'XDir','reverse'); % h decreases from left to right
hold off;

