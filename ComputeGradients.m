function [grad_W, grad_b] = ComputeGradients(X, Y, P, W,b, lambda)
g = -(Y-P)';
n = size(X,2);
grad_b2=(sum(g,1)/n)';
s1 = W{1}*X+repmat(b{1},1,n);
h = max(0, s1);
grad_W2 = g'*h'/n+2*lambda*W{2};
g=g*W{2};
g=g.*(s1>0)'; %ReLu
grad_b1=(sum(g,1)/n)';
grad_W1 = g'*X'/n+2*lambda*W{1};
grad_W={grad_W1,grad_W2};
grad_b={grad_b1,grad_b2};
end