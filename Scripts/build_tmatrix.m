function T = build_tmatrix(model,dT,bend)
%BUILD_TMATRIX Build P-by-N intensity transmission matrix.
T = zeros(model.P,model.N);
for k = 1:model.N
    T(:,k) = field_intensity(model,model.lam(k),dT,bend);
end
end
