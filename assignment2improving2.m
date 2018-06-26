%% 5.1 improving
clc;
clear;
addpath Datasets/cifar-10-matlab/cifar-10-batches-mat/;
%1.Read in the data & initialize the parameters 
%read data
%Use all the available training data for training (all five batches minus a small
%subset of the training images for a validation set). Decrease the size of the
%validation set down to 1000
[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat'); % training data part1
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat'); % training data part2
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat'); % training data part3
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat'); % training data part4
[X5,Y5,y5] = LoadBatch('data_batch_5.mat'); % training data part5

Xtrain5=X5(:,1:size(X5,2)-1000);
Xtrain=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
Ytrain5=Y5(:,1:size(Y5,2)-1000);
Ytrain=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
ytrain5=y5(:,1:size(X5,2)-1000);
ytrain=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];

Xvalid=X5(:,(size(X5,2)-999):size(X5,2));
Yvalid=Y5(:,(size(Y5,2)-999):size(Y5,2));
yvalid=y5(:,(size(y5,2)-999):size(y5,2));

[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data

mean_X = mean(Xtrain, 2);
Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid - repmat(mean_X, [1, size(Xvalid, 2)]);
Xtest = Xtest - repmat(mean_X, [1, size(Xtest, 2)]);
%initialization
% d=3072;%dimention
% n=10000;
% Xtrain=Xtrain(1:d,1:n);
% ytrain=ytrain(1:n);
% Ytrain=Ytrain(:,1:n);
% Xvalid=Xvalid(1:d,1:n);
% yvalid=yvalid(1:n);
% Yvalid=Yvalid(:,1:n);
% Xtest=Xtest(1:d,1:n);
% ytest=ytest(1:n);
% Ytest=Ytest(:,1:n);
[W,b]=initializae(Xtrain);  

%3. momentum
%mini-batch gradient descent
n_batch = 100; %the number of images in a mini-batch
n_epochs = 100; %the number of runs through the whole training set
rho=0.9; %momentum parameter:{0.5,0.9,0.99}
lambda=0.003475;
eta=0.04529;
GDparams = [n_batch, eta, n_epochs, rho];
[Wstar_t, bstar_t, J] = MiniBatchGDmo(Xtrain, Ytrain,GDparams, W, b, lambda);
%accuracy
accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
acc_valid = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);

%4.Training your network
%Draw loss picture
% figure();
% plot3(lambda,eta,acc_valid);
% hold on;
% hold off;
% grid on;
% xlabel('lambda');
% ylabel('eta');

%% 5.2 leaky ReLu
clc;
clear;
addpath Datasets/cifar-10-matlab/cifar-10-batches-mat/;

[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat'); % training data part1
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat'); % training data part2
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat'); % training data part3
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat'); % training data part4
[X5,Y5,y5] = LoadBatch('data_batch_5.mat'); % training data part5

Xtrain5=X5(:,1:size(X5,2)-1000);
Xtrain=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
Ytrain5=Y5(:,1:size(Y5,2)-1000);
Ytrain=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
ytrain5=y5(:,1:size(X5,2)-1000);
ytrain=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];

Xvalid=X5(:,(size(X5,2)-999):size(X5,2));
Yvalid=Y5(:,(size(Y5,2)-999):size(Y5,2));
yvalid=y5(:,(size(y5,2)-999):size(y5,2));

[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data

mean_X = mean(Xtrain, 2);
Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid - repmat(mean_X, [1, size(Xvalid, 2)]);
Xtest = Xtest - repmat(mean_X, [1, size(Xtest, 2)]);

[W,b]=initializae(Xtrain);  

%3. momentum
%mini-batch gradient descent
n_batch = 100; %the number of images in a mini-batch
n_epochs = 100; %the number of runs through the whole training set
rho=0.9; %momentum parameter:{0.5,0.9,0.99}
lambda=0.003475;
eta=0.04529;
GDparams = [n_batch, eta, n_epochs, rho];
[Wstar_t, bstar_t, J] = LeakyMiniBatchGDmo(Xtrain, Ytrain,GDparams, W, b, lambda);
%accuracy
accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
acc_valid = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);

%% functions for both
function [X, Y, y] = LoadBatch(filename)
 A = load(filename);
 X = double(A.data)/double(255); %normalized to figures between 0 and 1
 %X is of type "double"
 y = A.labels;
 [a,~] = size(y);
 K = 10;
 Y = zeros(a,K);
 for i = 1:a
 Y(i,y(i)+1) = 1;
 end
 X = X';
 Y = Y';
 y = y';
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~,n] = size(y);
correct = 0;
for i = 1:n
    [~,k(i)] = max(P(:,i));
    if y(i)+1 == k(i)
          correct = correct+1;
    end
end
acc = correct/n;
end

function [W,b]=initializae(X)
K = 10;
d = size(X,1); %dimention, all is 3072
m = 200;
rng(400);
% sigma=0.01;
sigma=sqrt(2/210); %He initialization
W1 = sigma*randn([m d]); % mean is 0, standard deviation is 0.001
W2 = sigma*randn([K m]);
b1 = sigma*randn([m 1]);
b2 = sigma*randn([K 1]);
W={W1,W2};
b={b1,b2};
end
%% functions for 5.1 
function P = EvaluateClassifier(X, W, b)
n = size(X,2);
W1 = W{1};
W2 = W{2};
b1 = b{1};
b1 = repmat(b1,1,n);
b2 = b{2};
b2 = repmat(b2,1,n);
s1 = W1*X+b1;
h = max(0, s1);
s = W2*h+b2;
P = softmax(s);
end

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2) + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
end

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

function [Wstar, bstar, J] = MiniBatchGDmo(X, Y,GDparams, W, b, lambda)
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

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);
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
    J(i) = ComputeCost(X, Y, W, b, lambda);
%     GDparams(2)=GDparams(2)*0.95; 
    if mod(i, 10) == 0 %This is add for bonus point 1
        GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
    end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end
%% functions for 5.2 
function P = LeakyEvaluateClassifier(X, W, b)
n = size(X,2);
W1 = W{1};
W2 = W{2};
b1 = b{1};
b1 = repmat(b1,1,n);
b2 = b{2};
b2 = repmat(b2,1,n);
s1 = W1*X+b1;
h = max(0.01*s1, s1);
s = W2*h+b2;
P = softmax(s);
end

function J = LeakyComputeCost(X, Y, W, b, lambda)
P = LeakyEvaluateClassifier(X, W, b);
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2) + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
end

function [grad_W, grad_b] = LeakyComputeGradients(X, Y, P, W,b, lambda)
g = -(Y-P)';
n = size(X,2);
grad_b2=(sum(g,1)/n)';
s1 = W{1}*X+repmat(b{1},1,n);
h = max(0.01*s1, s1);
grad_W2 = g'*h'/n+2*lambda*W{2};
g=g*W{2};
A = 0.01*ones(size(s1));
A(s1>0)=1;
g=g.*A'; %ReLu
grad_b1=(sum(g,1)/n)';
grad_W1 = g'*X'/n+2*lambda*W{1};
grad_W={grad_W1,grad_W2};
grad_b={grad_b1,grad_b2};
end

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