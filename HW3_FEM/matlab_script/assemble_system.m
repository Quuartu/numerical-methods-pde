function [H, A_nodal] = assemble_system(nodes, elems)
% ASSEMBLE_SYSTEM from Section 3.2 and 3.3 of the assignment text
% This function builds the global stiffness matrix H and computes nodal areas following the algorithm described in the assignment text.
%
%   INPUT:
%       nodes : Nx2 matrix containing (x,y) coordinates of vertices.
%       elems : Mx3 matrix containing the topology (node indices for each triangle).
%
%   OUTPUT:
%       H       : Global sparse stiffness matrix (NxN).
%       A_nodal : Nx1 vector containing the lumped area for each node.

    % Get mesh dimensions
    n_nodes = size(nodes, 1);
    n_elems = size(elems, 1);

    % =========================================================================
    % INITIALIZATION (Step 'a' in Algorithm, assignment text)
    % =========================================================================
    % We initialize the global matrix H with zeros. But since we know that H is sparse, we use spalloc for efficiency.
    % Thanks to spalloc, we preallocate space for only 7 non-zero entries per row (typical for 2D FEM with linear triangles).
    H = spalloc(n_nodes, n_nodes, 7 * n_nodes); 
    
    % Initialize nodal areas vector with zeros
    A_nodal = zeros(n_nodes, 1);

    % =========================================================================
    % LOOP OVER ELEMENTS (Step 'b' in Algorithm, assignment text)
    % =========================================================================
    for k = 1 : n_elems
        
        % --- A. Retrieve Coordinates ---
        % Get only the global index of the current triangle 'k'
        globe_idx = elems(k, :); % Indices [i, j, m]
        
        % Extract physical coordinates (x, y)
        x = nodes(globe_idx, 1);
        y = nodes(globe_idx, 2);
        
        % --- B. Compute Gradients and Area (Sec 3.2) ---
        % Compute geometric coefficients 'b' (partial derivative w.r.t x related)
        % b_1 = y2 - y3; b_2 = y3 - y1; b_3 = y1 - y2
        b = [y(2) - y(3); 
             y(3) - y(1); 
             y(1) - y(2)];
             
        % Compute geometric coefficients 'c' (partial derivative w.r.t y related)
        % c_1 = x3 - x2; c_2 = x1 - x3; c_3 = x2 - x1
        c = [x(3) - x(2); 
             x(1) - x(3); 
             x(2) - x(1)];
        
        % Compute the area of the triangle (Determinant / 2)
        % Using the formula: Area = 0.5 * det([1 x1 y1; ...])
        Delta = 0.5 * det([ones(3,1), x, y]);
        
        % --- C. Compute Local Stiffness Matrix (Sec 3.2) ---
        % The local matrix H_loc is 3x3 for each triangle
        H_loc = (1 / (4 * Delta)) * (b * b' + c * c');
        
        % --- D. Nodal Area Lumping (Sec 3.3) ---
        % Accumulate 1/3 of the element area to each of its vertices. This is required for the L2 error norm computation later. In fact we need to divide area by 3 for each vertex so every node gets its share of the area.
        A_nodal(globe_idx) = A_nodal(globe_idx) + Delta / 3;
        
        % =====================================================================
        % 3. ASSEMBLY INTO GLOBAL MATRIX (Step 'b' - inner loops)
        % =====================================================================
        % We map the local 3x3 matrix entries into the global NxN matrix. We takes values from H_loc and add them to the appropriate positions in H according to the global node indices. '+' is perfectly fine here since we are accumulating contributions from multiple elements.
        % H(R,C) = H(R,C) + H_loc(r,c)
        
        % Loop over local rows (r_loc = 1, 2, 3)
        for r_loc = 1 : 3
            % Identify global Row index R
            row_global = globe_idx(r_loc);
            
            % Loop over local columns (c_loc = 1, 2, 3)
            for c_loc = 1 : 3
                % Identify global Column index C
                col_global = globe_idx(c_loc);
                
                % Update the global matrix (Accumulation)
                H(row_global, col_global) = H(row_global, col_global) + H_loc(r_loc, c_loc);
            end
        end
        
    end
    % End of loop over elements

end