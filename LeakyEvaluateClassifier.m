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