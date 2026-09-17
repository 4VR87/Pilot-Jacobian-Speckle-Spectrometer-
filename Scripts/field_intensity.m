function I = field_intensity(model,lambda_nm,dT,bend)
%FIELD_INTENSITY Normalized camera speckle fingerprint at one wavelength.
dl = lambda_nm - model.lam0;
ph = model.phi0 + model.s_lam.*dl + 0.5*model.q_lam.*(dl.^2) + ...
     model.alphaT.*dT + model.alphaB.*bend + model.alphaTB.*dT.*bend;
v = model.b0 .* exp(1i*ph);
Ae = model.A + dT.*model.BT + bend.*model.BB;
E = Ae*v;
I = abs(E).^2;
I = I ./ (sum(I) + eps);
end
