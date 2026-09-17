function x = reconstruct_spectrum(model,T,y,alpha,beta)
%RECONSTRUCT_SPECTRUM Ridge + first-difference smoothness, then nonnegative clipping.
if nargin < 4 || isempty(alpha), alpha = model.alphaReg; end
if nargin < 5 || isempty(beta),  beta  = model.betaReg; end
H = T'*T + alpha*eye(model.N) + beta*(model.D'*model.D);
x = H \ (T'*y);
x = max(x,0);
mx = max(x);
if mx > 0, x = x./mx; end
end
