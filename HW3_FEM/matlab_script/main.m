clear all; close all; clc;

% =========================================================================
% MAIN SCRIPT: FEM Solution of Poisson Equation
% Implements the workflow described in "Figure 2: Flowchart" (Page 4)
% =========================================================================

% Configuration
n_levels = 5;      % Meshes from 0 to 4 (Requirements, Page 11)
tol = 1e-8;        % PCG Tolerance (Section 3.5, Page 8)
max_it = 2000;     % Safety limit for iterations
results = [];      % Data structure for storage

% =========================================================================
% LOOP OVER REFINEMENT LEVELS (0 -> 4)
% =========================================================================
for k = 0 : n_levels-1
    fprintf('Processing Mesh Level %d...\n', k);
    
    % ---------------------------------------------------------------------
    % 1. INPUT DATA (Section 3.1)
    % ---------------------------------------------------------------------
    folder_name = sprintf('mesh%d', k); 
    file_name   = sprintf('mesh%d', k);
    base_path   = fullfile('inputfiles', folder_name, file_name);
    
    % Load topology (.topol), coordinates (.coord), boundary (.bound)
    if ~isfile([base_path '.coord'])
        error('File not found: %s', [base_path '.coord']);
    end
    nodes = load([base_path '.coord']);       
    elems = load([base_path '.topol']);       
    dirichlet = load([base_path '.bound']);   
    
    % Load trace indices if available (for Section 4 validation)
    trace_idx = [];
    if isfile([base_path '.trace'])
        trace_idx = load([base_path '.trace']);
    end
    
    % Compute Mesh Size h (max edge length) - Section 3.6
    h_max = 0;
    for e = 1:size(elems, 1)            % Search all triangles
        ids = elems(e, :);              % Node indices of triangle found
        pts = nodes(ids, :);
        % Calculate edge lengths of triangle by Pythagoras
        edges = [norm(pts(1,:)-pts(2,:)), ...     
                 norm(pts(2,:)-pts(3,:)), ...
                 norm(pts(3,:)-pts(1,:))];
        % Update global max edge length
        h_max = max(h_max, max(edges));
    end
    
    % ---------------------------------------------------------------------
    % 2. ASSEMBLY (Section 3.2 & Alg 3.1) Code of this function is in
    %    'assemble_system.m'
    % ---------------------------------------------------------------------
    [H, A_nodal] = assemble_system(nodes, elems);       
    
    % ---------------------------------------------------------------------
    % 3. BOUNDARY CONDITIONS (Section 3.4 & Alg 3.2) Code of this function is
    %    in 'apply_boundary_conditions.m'
    % ---------------------------------------------------------------------
    n_nodes = size(nodes, 1);
    f = zeros(n_nodes, 1); % Homogeneous source term f=0
    [H_mod, f_mod] = apply_boundary_conditions(H, f, dirichlet);
    
    % ---------------------------------------------------------------------
    % 4. LINEAR SYSTEM SOLUTION (Section 3.5)
    % ---------------------------------------------------------------------
    % A) Jacobi Preconditioner
    fprintf('  Solving with Jacobi...\n');
    M_jac = diag(diag(H_mod));                      % Function of Jacobi Preconditioner
    [~, ~, ~, ~, resvec_jac] = pcg(H_mod, f_mod, tol, max_it, M_jac);
    
    % B) Incomplete Cholesky (ICHOL) Preconditioner ('nofill')
    fprintf('  Solving with ICHOL...\n');
    opts.type = 'nofill'; 
    L_ichol = ichol(H_mod, opts);                   % Function of ICHOL Preconditioner
    [u_h, ~, ~, ~, resvec_ichol] = pcg(H_mod, f_mod, tol, max_it, L_ichol, L_ichol');
    
    % ---------------------------------------------------------------------
    % 5. SAVE RESULTS
    % ---------------------------------------------------------------------
    results(k+1).u = u_h;
    results(k+1).nodes = nodes;
    results(k+1).h_size = h_max;
    results(k+1).A_nodal = A_nodal;
    results(k+1).resvec_jac = resvec_jac;
    results(k+1).resvec_ichol = resvec_ichol;
    results(k+1).trace_idx = trace_idx; 
    
    % Plot 3D Solution (Req. 1, Page 11)
    figure(100 + k);
    trisurf(elems, nodes(:,1), nodes(:,2), u_h);
    title(['Solution u_h - Level ' num2str(k) ' (h=' num2str(h_max, '%.4f') ')']);
    xlabel('x'); ylabel('y'); zlabel('u');
    shading interp; view(-45, 30); axis equal; zlim([-0.1 1.1]); 
    drawnow;
end

% =========================================================================
% ERROR COMPUTATION (Section 3.6)
% =========================================================================
fprintf('\n=== FEM CONVERGENCE TABLE (Eq. 19) ===\n');
fprintf('Lev | h_size   | Error L2     | Ratio r_k\n');
fprintf('-----------------------------------------\n');

% 1. LOAD REFERENCE DATA
ref_file_name = 'solRef.dat'; 
ref_path = fullfile('inputfiles', ref_file_name);

if isfile(ref_path)
    % Load the ASCII file (Format: Column 1=x, Column 2=y, Column 3=u)
    ref_data = load(ref_path);
    
    xRef = ref_data(:, 1);
    yRef = ref_data(:, 2);
    uRef = ref_data(:, 3);
    
    % 2. CREATE INTERPOLANT (Algorithm 3.3)
    % Creates a function F(x,y) = u based on the dense reference data
    % explicitly requested for non-nested meshes.
    ref_interpolant = scatteredInterpolant(xRef, yRef, uRef);
    
else
    error('Reference file not found at: %s. Please check the filename.', ref_path);
end

% 3. COMPUTE ERROR ON ALL LEVELS
% We iterate from k=1 (Mesh 0) to k=n_levels (Mesh 4)
for k = 1 : n_levels 
    
    % Extract current mesh nodes
    curr_nodes = results(k).nodes;
    
    % Interpolate the high-resolution reference solution on current nodes
    u_ref_mapped = ref_interpolant(curr_nodes(:,1), curr_nodes(:,2));
    
    % Compute L2 Error Norm (Implementation of Eq. 18)
    % Error = sqrt( sum( (u_h - u_ref)^2 * Area_i ) )
    diff = results(k).u - u_ref_mapped;
    weights = results(k).A_nodal; % Nodal areas from assembly
    error_L2 = sqrt(sum((diff.^2) .* weights));
    
    % Store error for plotting or further analysis
    results(k).error_L2 = error_L2;
    
    % Compute Convergence Ratio r_k (Implementation of Eq. 19)
    % r_k = (e_prev / e_curr) * (h_curr / h_prev)^2
    ratio_print = 0;
    if k > 1
        err_prev = results(k-1).error_L2;
        h_prev   = results(k-1).h_size;
        h_curr   = results(k).h_size;
        
        % Avoid division by zero if error is extremely small (e.g. on Mesh 4)
        if error_L2 > 1e-15
            ratio_print = (err_prev / error_L2) * (h_curr / h_prev)^2;
        else
            ratio_print = NaN; % Reference mesh reached (Error ~ 0)
        end
    end
    
    % Print row in the table
    fprintf('  %d | %.4f   | %.4e   | %.4f \n', k-1, results(k).h_size, error_L2, ratio_print);
end

% =========================================================================
% PLOTS: SOLVER CONVERGENCE (Req. 2, Page 11)
% =========================================================================
% These plots use semi-logarithmic scale as requested for better visibility. 

% Figure 201: JACOBI
figure(201); clf; hold on;
legend_str = {};
for k = 1:n_levels
    rv = results(k).resvec_jac;
    % Plot normalized residual vs iterations (Semi-Log Scale)
    semilogy(0:length(rv)-1, rv/rv(1), 'LineWidth', 1.5);
    legend_str{end+1} = ['Mesh ' num2str(k-1)];
end
title('PCG Convergence: JACOBI'); 
xlabel('Iterations'); ylabel('Relative Residual (log scale)');
grid on; legend(legend_str, 'Location', 'Best');
set(gca, 'YScale', 'log'); % Force Log Scale

% Figure 202: ICHOL
figure(202); clf; hold on;
legend_str = {};
for k = 1:n_levels
    rv = results(k).resvec_ichol;
    % Plot normalized residual vs iterations (Semi-Log Scale)
    semilogy(0:length(rv)-1, rv/rv(1), 'LineWidth', 1.5);
    legend_str{end+1} = ['Mesh ' num2str(k-1)];
end
title('PCG Convergence: ICHOL'); 
xlabel('Iterations'); ylabel('Relative Residual (log scale)');
grid on; legend(legend_str, 'Location', 'Best');
set(gca, 'YScale', 'log'); % Force Log Scale

% =========================================================================
% PLOT: TRACE VALIDATION (Section 4, Results) 
% =========================================================================
if ~isempty(results(end).trace_idx)
    figure(300); clf;
    
    trace_data = results(end).trace_idx;
    % Robust check for columns
    if size(trace_data, 2) >= 2
        t_idx = trace_data(:, 1); arc_len = trace_data(:, 2);
    else
        t_idx = trace_data; arc_len = [];
    end
    
    t_idx = round(t_idx);
    if min(t_idx) == 0, t_idx = t_idx + 1; end
    
    % Extract solution
    u_t = results(end).u(t_idx);
    
    if ~isempty(arc_len)
        plot(arc_len, u_t, 'w-', 'LineWidth', 2); 
    else
        plot(u_t, 'w-', 'LineWidth', 2);
    end
    title('Validation: Solution along \Gamma_N'); xlabel('Arc Length'); ylabel('u'); 
    grid on; axis tight;
    fprintf('Trace Plot generated.\n');
end