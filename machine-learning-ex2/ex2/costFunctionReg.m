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



%sum = 0;

%for index = 1:m
%	yVal = y(index,1);
%	xi = X(index:index,:);
%	Z = xi * theta;
%	hThetaXi = sigmoid(Z);
%	sum = sum + ((-yVal * log(hThetaXi)) - ((1 - yVal)*log(1 - hThetaXi)));	
%end

%regul = 0;
%n = length(theta);
%for index = 2:n
%	regul = regul + (theta(index) .^ 2);
%end

%regul = (lambda * regul) / (2*m);

%J = sum/m + regul;

%grad = ((X' * (sigmoid(X * theta) - y)) + (lambda * theta)) / m;
%disp(grad(1));
%temp_grad = (X' * (sigmoid(X * theta) - y)) / m;
%grad(1) = temp_grad(1);


% ================
%z = X * theta;
%ht = sigmoid(z);
%total = (-1 * (y' * log(ht))) - ((1 - y)' * log(1 - ht));
%s = total/m;
%t = (X' * (ht - y)) ./ m ;
[s, t] = costFunction(theta, X, y);

ot = theta;
theta(1) = 0;
tsum = sum(theta .^ 2);
tsum = tsum - theta(1);
reg = (lambda / (2 * m)) * tsum;
J = s + reg;

reg = ot .+ (lambda / m);
grad = t + reg;
grad(1) = t(1); %replace the first value

% busy with office work today!!

% =============================================================

end
