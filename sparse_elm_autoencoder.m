function x = sparse_elm_autoencoder(A,b,lam,itrs)

% This is adapted from the code provided by [1].
% [1] J. Tang, C. Deng and G. Huang, "Extreme Learning Machine for Multilayer Perceptron," in IEEE Transactions on Neural Networks and Learning Systems, vol. 27, no. 4, pp. 809-821, April 2016.

AA = (A') * A;

Lf = max(eig(AA));
Li = 1/Lf;
alp = lam * Li;
m = size(A,2);
n = size(b,2);
x = zeros(m,n);
yk = x;
tk = 1;
L1 = 2 * Li * AA;
L2 = 2 * Li * A' * b;
% tic
for i = 1:itrs,
  ck = yk - L1*yk + L2;
  x1 = (max(abs(ck)-alp,0)).*sign(ck);
  tk1 = 0.5 + 0.5*sqrt(1+4*tk^2);
  tt = (tk-1)/tk1;
  yk = x1 + tt*(x-x1);
  tk = tk1;x = max(x1,0);
end
% toc

