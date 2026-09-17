function Tup = single_pilot_update(model,dT,bend,pilot_snr_db)
%SINGLE_PILOT_UPDATE Temperature-like one-dimensional correction using first pilot.
Gp = model.G(1:model.P,1);
y0p = model.y0pil(1:model.P);
y = add_noise_snr(field_intensity(model,model.pilot_lams(1),dT,bend),pilot_snr_db);
qt = (Gp'*Gp + 1e-10) \ (Gp'*(y-y0p));
Tup = model.T0 + model.JT.*qt;
end
