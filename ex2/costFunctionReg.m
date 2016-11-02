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

J_tmp = 0;
grad_tmp = zeros(size(theta));

[J_tmp, grad_tmp] = costFunction(theta, X, y);

theta_tmp = theta;
theta_tmp(1) = 0;
J = J_tmp + (theta_tmp'*theta_tmp)*lambda/(2*m);

grad2 = (lambda/m).*theta;
grad2(1)=0;
grad = grad_tmp + grad2;


% =============================================================

end
