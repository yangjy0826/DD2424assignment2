function [Wstar, bstar, J] = LeakyMiniBatchGDmo(X, Y,GDparams, W, b, lambda)
N = size(X,2);
v_W1=0;
v_W2=0;
v_b1=0;
v_b2=0;

for i=1:GDparams(3)
    for j=1:N/GDparams(1)
        j_start = (j-1)*GDparams(1) + 1;
        j_end = j*GDparams(1);
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = LeakyEvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = LeakyComputeGradients(Xbatch, Ybatch, P, W, b, lambda);
        %with momentum 
        v_W1=GDparams(4)*v_W1+GDparams(2)*grad_W{1};
        W{1} = W{1} - v_W1;
        v_W2=GDparams(4)*v_W2+GDparams(2)*grad_W{2};
        W{2} = W{2} - v_W2;
        v_b1=GDparams(4)*v_b1+GDparams(2)*grad_b{1};
        b{1} = b{1} - v_b1;
        v_b2=GDparams(4)*v_b2+GDparams(2)*grad_b{2};
        b{2} = b{2} - v_b2;
    end
    J(i) = LeakyComputeCost(X, Y, W, b, lambda);
%     GDparams(2)=GDparams(2)*0.95; 
    if mod(i, 10) == 0 %This is add for bonus point 1
        GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
    end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end