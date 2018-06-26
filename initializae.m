function [W,b]=initializae(X)
K = 10;
d = size(X,1); %dimention, all is 3072
m = 200;
% rng(400);
% sigma=0.01;
sigma=sqrt(2/210); %He initialization
W1 = sigma*randn([m d]); % mean is 0, standard deviation is 0.001
W2 = sigma*randn([K m]);
b1 = sigma*randn([m 1]);
b2 = sigma*randn([K 1]);
W={W1,W2};
b={b1,b2};
end