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


%grad = (1/m)*(X' * (sigmoid(X * theta) - y))+((lambda/m).*(theta(2:end)));
%gradtheta0 = (1/m)*sum(X' * (sigmoid(X * theta) - y));
%gradall = (lambda/m).*(theta(2:end));
%gradall = [0; gradall];
%grad = gradtheta0+gradall;
%((lambda/m)*sum(theta(setdiff(1:end,1) )))


gradnormal = (1/m)*(X' * (sigmoid(X * theta) - y));
%size(gradnormal)
gradregularized = (lambda/m) .* theta(2:end);
%size(gradregularized)
gradregularized = [0; gradregularized];
%size(gradregularized)
grad = gradnormal + gradregularized;
%size(grad)
J = ((1/m)*sum((-y .* log(sigmoid(X * theta))) - ((1 - y) .* log(1 - sigmoid(X * theta))))) + (lambda/(2*m))*sum(theta(setdiff(1:end,1)).^2);



% =============================================================

end
