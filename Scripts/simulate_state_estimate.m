function qhat = simulate_state_estimate(model,dT,bend,pilot_snr_db,jacobian_fractional_std)
%SIMULATE_STATE_ESTIMATE Acquire pilots and estimate [temperature; bend].
if nargin < 5, jacobian_fractional_std = 0; end
y = acquire_pilots(model,dT,bend,pilot_snr_db);
qhat = estimate_state_from_pilots(model,y,jacobian_fractional_std);
end
