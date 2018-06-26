function [Wstar, bstar, J] = MiniBatchGD(X, Y,GDparams, W, b, lambda)
N = size(X,2);

for i=1:GDparams(3)
    for j=1:N/GDparams(1)
        j_start = (j-1)*GDparams(1) + 1;
        j_end = j*GDparams(1);
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);

%         without momentum
        W{1} = W{1} - GDparams(2) * grad_W{1};
        W{2} = W{2} - GDparams(2) * grad_W{2};
        b{1} = b{1} - GDparams(2) * grad_b{1};
        b{2} = b{2} - GDparams(2) * grad_b{2};
    end
    J(i) = ComputeCost(X, Y, W, b, lambda);
    GDparams(2)=GDparams(2)*0.95; 
end
Wstar = W;
bstar = b;
end