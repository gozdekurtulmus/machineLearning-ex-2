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
root=0;
ho = sigmoid(X*theta);
for i=1:m
  root = root + (-y(i))*log(ho(i)) - (1-y(i))*log(1-ho(i));
endfor
J = 1/m .* root;


for a=1:length(theta)
  root0 = 0;
  for i=1:m
    root0 = root0 + (ho(i)- y(i))*X(i,a) ;
  grad(a) = root0/m;
  endfor
endfor



% =============================================================

end
