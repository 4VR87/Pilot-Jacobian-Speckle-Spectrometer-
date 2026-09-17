function y = acquire_pilots(model,dT,bend,pilot_snr_db)
%ACQUIRE_PILOTS Simulate the two time-gated pilot frames.
y = zeros(2*model.P,1);
for p = 1:2
    idx = model.pilot_idx(p);
    v = field_intensity(model,model.lam(idx),dT,bend);
    y((p-1)*model.P+(1:model.P)) = add_noise_snr(v,pilot_snr_db);
end
end
