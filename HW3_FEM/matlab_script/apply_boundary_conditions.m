function [H, f] = apply_boundary_conditions(H, f, dirichlet)
% APPLY_BOUNDARY_CONDITIONS Enforces Dirichlet boundary conditions.
% Implementation of "Algorithm 3.2" (Page 9 of FEM_exercise.pdf).
%
% This function modifies the linear system Hu=f to enforce u = u_D on 
% the Dirichlet boundary, while preserving the Symmetry of H.
%
%   INPUT:
%       H         : Global stiffness matrix (NxN)
%       f         : Global RHS vector (Nx1)
%       dirichlet : Kx2 matrix [node_index, imposed_value]
%
%   OUTPUT:
%       H, f      : Modified system enforcing BCs

    nodes_idx = dirichlet(:, 1); 
    values    = dirichlet(:, 2); 
    
    % =========================================================================
    % STEP 1: RHS UPDATE (Corresponds to Alg 3.2, Line 2)
    % =========================================================================
    % "Update rhs with to-be-removed columns"
    % We subtract the known terms H_{ji} * u_{D,i} from the RHS.
    % Eq. 13 in the text suggests moving known terms to the right.
    for k = 1:length(nodes_idx)
        idx = nodes_idx(k);
        val = values(k);
        
        if val ~= 0
            % f = f - H(:, col) * val
            f = f - H(:, idx) * val; 
        end
    end
    
    % =========================================================================
    % STEP 2: MATRIX ZEROING (Corresponds to Alg 3.2, Lines 1 & 4)
    % =========================================================================
    % "Set all the Dirichlet rows to zero" (Line 1)
    % "Actually remove the Dirichlet columns" (Line 4)
    
    % We perform this simultaneously for efficiency:
    H(:, nodes_idx) = 0; % Zero columns
    H(nodes_idx, :) = 0; % Zero rows
    
    % =========================================================================
    % STEP 3: DIAGONAL FORCING (Corresponds to Alg 3.2, Lines 3 & 6)
    % =========================================================================
    % "Set the rhs values on the Dirichlet rows" (Line 3)
    % "Finally set diagonal term to 1" (Line 6)
    
    for k = 1:length(nodes_idx)
        idx = nodes_idx(k);
        val = values(k);
        
        H(idx, idx) = 1.0; % Restore diagonal
        f(idx) = val;      % Enforce value on RHS
    end

end