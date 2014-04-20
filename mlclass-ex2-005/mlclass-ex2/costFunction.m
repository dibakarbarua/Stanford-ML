function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = zeros(m,1);

for i = 1:1:m
    h(i) = transpose(theta) * transpose(X(i,:));
end

for i = 1:1:m
    h(i) = sigmoid(h(i));
end

J_vector = zeros(m,1);

J_vector = -1 .* ( (y .* log(h)) + (ones(m,1) - y).*(log(ones(m,1) - h)) );

J = sum(J_vector,1)/m;

grad_temp = zeros(m,1);

for j = 1:1:size(theta)
    for i = 1:1:m
        grad_temp(i,1) = (h(i) - y(i)) * X(i,j);
    end
    grad(j) = sum(grad_temp,1)/m;
end








% =============================================================

end
