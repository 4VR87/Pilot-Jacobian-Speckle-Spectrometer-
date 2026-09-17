function validate_reference_model(model)
%VALIDATE_REFERENCE_MODEL Check the MATLAB forward model against stored reference matrices.
Tcheck = build_tmatrix(model,0,0);
errT0 = max(abs(Tcheck(:)-model.T0(:)));
JTcheck = (build_tmatrix(model,model.hT,0)-build_tmatrix(model,-model.hT,0))/(2*model.hT);
JBcheck = (build_tmatrix(model,0,model.hB)-build_tmatrix(model,0,-model.hB))/(2*model.hB);
errJT = max(abs(JTcheck(:)-model.JT(:)));
errJB = max(abs(JBcheck(:)-model.JB(:)));
Gcheck = [vertcat(JTcheck(:,model.pilot_idx(1)),JTcheck(:,model.pilot_idx(2))), ...
          vertcat(JBcheck(:,model.pilot_idx(1)),JBcheck(:,model.pilot_idx(2)))];
errG = max(abs(Gcheck(:)-model.G(:)));
fprintf('max|T0_MATLAB - T0_reference| = %.3e\n',errT0);
fprintf('max|JT_MATLAB - JT_reference| = %.3e\n',errJT);
fprintf('max|JB_MATLAB - JB_reference| = %.3e\n',errJB);
fprintf('max|G_MATLAB  - G_reference|  = %.3e\n',errG);
fprintf('cond(G) MATLAB = %.12g ; reference = %.12g\n',cond(Gcheck),model.Gcond);
if max([errT0,errJT,errJB,errG]) > 1e-10
    warning('Deterministic translation differs from the stored reference above 1e-10.');
end
end
