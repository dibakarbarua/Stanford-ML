function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

J_extra = zeros((size(theta)-1),1);

for i=1:1:(size(theta)-1)
    J_extra(i) = theta(i+1) * theta(i+1);
end
   
J = J + (lambda * (sum(J_extra,1)))/(2*m);

grad_temp = zeros(m,1);

for j = 1:1:size(theta)
    for i = 1:1:m
        grad_temp(i,1) = (h(i) - y(i)) * X(i,j);
    end
    grad(j) = sum(grad_temp,1)/m;
    if(j>1)
        grad(j) = grad(j) + lambda * theta(j)/m;
    end
end





% =============================================================

end
