function qhat = estimate_state_from_pilots(model,y,jacobian_fractional_std)
%ESTIMATE_STATE_FROM_PILOTS Regularized least-squares two-state estimate.
if nargin < 3, jacobian_fractional_std = 0; end
Guse = model.G;
if jacobian_fractional_std > 0
    Guse = Guse .* (1 + jacobian_fractional_std.*randn(size(Guse)));
end
qhat = (Guse'*Guse + 1e-10*eye(2)) \ (Guse'*(y-model.y0pil));
end
