function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = zeros(m,1);

tempval = zeros(1,m);
tempval_t = zeros(m,1);
theta_t = zeros(1,2);
X_t = zeros(2,m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

theta_t = transpose(theta);
X_t = transpose(X);
tempval = theta_t * X_t; % 1 * m vector
tempval_t = transpose(tempval); % m * 1vector
tempval_t = tempval_t - y;
tempval_t = tempval_t .* tempval_t;

J = sum(tempval_t,1);
J = J/(2*m);




% =========================================================================

end
