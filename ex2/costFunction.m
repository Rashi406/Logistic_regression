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

c=theta' * X';
c=c';
h=sigmoid(c);
a=log(h);
b=log(1-h);
s=(y.*a)+((1-y).*b);
s=sum(s);
J=(-(1/m))*s;
a= (h-y);
n=size(X,2);
for i=1:n
del(:,i)=a.*X(:,i);
endfor
delta=(1/m)*sum(del);
grad=delta';






% =============================================================

end
