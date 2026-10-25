% Solution for the exercise on the 4-stage Runge-Kutta method

% Define the function for the ODE
f = @(t, y) -10 * y^2;

% Define the analytical solution
y_true = @(t) 1 ./ (10*t + 1);

% Parameters
T = 2;
y0 = 1;
k_values = 5:10;
num_k = length(k_values);

h_values = zeros(num_k, 1);
errors = zeros(num_k, 1);
n_steps_values = zeros(num_k, 1);

% Main loop for different step sizes
for i = 1:num_k
    k = k_values(i);
    h = 2^(-k);
    h_values(i) = h;
    
    n_steps = T / h;
    n_steps_values(i) = n_steps;
    
    t = 0;
    y = y0;
    
    % RK4 method
    for j = 1:n_steps
        k1 = f(t, y);
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        k4 = f(t + h, y + h * k3);
        
        y = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4);
        t = t + h;
    end
    
    % Calculate error at T=2
    y_final_numerical = y;
    y_final_true = y_true(T);
    errors(i) = abs(y_final_numerical - y_final_true);
end

% Display the table
fprintf('k\t\th\t\t\tError\n');
fprintf('----------------------------------\n');
for i = 1:num_k
    fprintf('%d\t\t%.6f\t\t%.6e\n', k_values(i), h_values(i), errors(i));
end

% Create the loglog plot
figure;
loglog(n_steps_values, errors, '-o');
xlabel('Number of steps');
ylabel('Error at T=2');
title('Error of 4-stage Runge-Kutta method');
grid on;

% Check error reduction
fprintf('\nError reduction check:\n');
for i = 1:num_k-1
    ratio = errors(i) / errors(i+1);
    fprintf('Ratio of errors for h=%.6f and h=%.6f: %.2f\n', h_values(i), h_values(i+1), ratio);
end
fprintf('The theoretical ratio is 2^4 = 16.\n');
