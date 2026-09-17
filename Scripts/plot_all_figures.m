function plot_all_figures(model,dataDir,figDir)
%PLOT_ALL_FIGURES Generate 15 manuscript-style figures from MATLAB results.
if ~exist(figDir,'dir'), mkdir(figDir); end

% Fig. 1 architecture
f=figure('Color','w','Position',[100 100 1200 560]); ax=axes(f,'Position',[0 0 1 1]); axis(ax,[0 12 0 6]); axis(ax,'off');
rectangle(ax,'Position',[0.4 3.7 1.8 1.1],'Curvature',0.08); text(ax,1.3,4.25,'Unknown spectrum\n1547-1553 nm','HorizontalAlignment','center');
rectangle(ax,'Position',[0.4 1.6 1.8 1.1],'Curvature',0.08); text(ax,1.3,2.15,'Two gated pilots\n1546.5 / 1553.5 nm','HorizontalAlignment','center');
rectangle(ax,'Position',[2.8 2.7 1.6 1.3],'Curvature',0.08); text(ax,3.6,3.35,'3x1 coupler\n+ mode scrambler','HorizontalAlignment','center');
rectangle(ax,'Position',[5.0 2.6 1.8 1.5],'Curvature',0.08); text(ax,5.9,3.35,'2-m MMF\n105-um core\nNA = 0.22','HorizontalAlignment','center');
rectangle(ax,'Position',[7.4 2.6 1.8 1.5],'Curvature',0.08); text(ax,8.3,3.35,'InGaAs camera\n12x12 ROI','HorizontalAlignment','center');
rectangle(ax,'Position',[9.8 3.7 1.7 1.1],'Curvature',0.08); text(ax,10.65,4.25,'Dual-pilot\nstate estimator','HorizontalAlignment','center');
rectangle(ax,'Position',[9.8 1.7 1.7 1.1],'Curvature',0.08); text(ax,10.65,2.25,'Jacobian-updated\ntransmission matrix','HorizontalAlignment','center');
rectangle(ax,'Position',[9.8 .25 1.7 .9],'Curvature',0.08); text(ax,10.65,.70,'Nonnegative\nspectral inversion','HorizontalAlignment','center');
text(ax,6,5.55,'Pilot-Jacobian self-calibration: physical-state tracking without wavelength rescanning','HorizontalAlignment','center','FontWeight','bold');
annotation(f,'arrow',[.20 .26],[.70 .60]); annotation(f,'arrow',[.20 .26],[.36 .50]); annotation(f,'arrow',[.38 .45],[.56 .56]); annotation(f,'arrow',[.58 .65],[.56 .56]); annotation(f,'arrow',[.77 .82],[.60 .70]); annotation(f,'arrow',[.88 .88],[.62 .45]); annotation(f,'arrow',[.88 .88],[.31 .20]);
exportgraphics(f,fullfile(figDir,'Fig01_architecture.png'),'Resolution',300); close(f);

% Fig. 2 speckles
I0=reshape(field_intensity(model,1550,0,0),12,12); I1=reshape(field_intensity(model,1550,1.2,0.18),12,12);
f=figure('Color','w','Position',[100 100 1100 400]); tiledlayout(f,1,3,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(I0); axis image; colorbar; title('Nominal speckle'); xlabel('camera x'); ylabel('camera y');
nexttile; imagesc(I1); axis image; colorbar; title('Thermal + bend drift'); xlabel('camera x'); ylabel('camera y');
nexttile; imagesc(I1-I0); axis image; colorbar; title('Difference'); xlabel('camera x'); ylabel('camera y');
exportgraphics(f,fullfile(figDir,'Fig02_speckle_drift.png'),'Resolution',300); close(f);

% Fig. 3 processing
f=figure('Color','w','Position',[100 100 1100 480]); ax=axes(f,'Position',[0 0 1 1]); axis(ax,[0 11 0 5]); axis(ax,'off');
labels={'Pilot frames','Subtract nominal\npilot speckles','Solve 2x2\nstate inverse','Update full TM\nT-hat=T0+JT*dT+JB*b','Acquire sample\nspeckle','Nonnegative\nregularized inverse','Recovered spectrum\n+ state report'}; pos=[.3 3.2;2.2 3.2;4.3 3.2;6.4 3.2;.3 1.2;6.4 1.2;8.7 1.2];
for k=1:size(pos,1), rectangle(ax,'Position',[pos(k,1) pos(k,2) 1.6 .9],'Curvature',.08); text(ax,pos(k,1)+.8,pos(k,2)+.45,labels{k},'HorizontalAlignment','center'); end
exportgraphics(f,fullfile(figDir,'Fig03_processing.png'),'Resolution',300); close(f);

% Fig. 4
T=readtable(fullfile(dataDir,'spectral_correlation.csv')); f=figure('Color','w'); plot(T.wavelength_nm-1550,T.corr_to_1550,'LineWidth',1.4); grid on; xlabel('Wavelength offset (nm)'); ylabel('Speckle correlation'); exportgraphics(f,fullfile(figDir,'Fig04_spectral_correlation.png'),'Resolution',300); close(f);

% Fig. 5
TT=readtable(fullfile(dataDir,'thermal_correlation.csv')); BB=readtable(fullfile(dataDir,'bend_correlation.csv')); f=figure('Color','w'); plot(TT.dT_C,TT.corr,'LineWidth',1.4); hold on; plot(BB.curvature_equiv_m_1,BB.corr,'LineWidth',1.4); grid on; xlabel('Perturbation coordinate (deg C or m^{-1})'); ylabel('Speckle correlation'); legend('temperature','equiv. curvature','Location','best'); exportgraphics(f,fullfile(figDir,'Fig05_environment_correlation.png'),'Resolution',300); close(f);

% Fig. 6
f=figure('Color','w','Position',[100 100 1000 420]); tiledlayout(f,1,2,'Padding','compact'); nexttile; imagesc(model.T0); axis xy; xlabel('Wavelength channel'); ylabel('Camera pixel'); title('Nominal transmission matrix'); colorbar; nexttile; sv=svd(model.T0); semilogy(sv/sv(1),'LineWidth',1.3); grid on; xlabel('Index'); ylabel('Normalized singular value'); exportgraphics(f,fullfile(figDir,'Fig06_tm_singular.png'),'Resolution',300); close(f);

% Fig. 7
T=readtable(fullfile(dataDir,'pilot_state_estimation.csv')); f=figure('Color','w'); plot(T.pilot_snr_db,T.temperature_rmse_C,'-o','LineWidth',1.3); hold on; plot(T.pilot_snr_db,T.bend_state_rmse,'-s','LineWidth',1.3); grid on; xlabel('Pilot SNR (dB)'); ylabel('RMSE'); legend('Temperature RMSE (deg C)','Bend-state RMSE','Location','best'); exportgraphics(f,fullfile(figDir,'Fig07_state_estimation.png'),'Resolution',300); close(f);

% Fig. 8
T=readtable(fullfile(dataDir,'reconstruction_example.csv')); f=figure('Color','w'); plot(T.wavelength_nm,T.true,'LineWidth',1.5); hold on; plot(T.wavelength_nm,T.stale); plot(T.wavelength_nm,T.single_pilot); plot(T.wavelength_nm,T.proposed); plot(T.wavelength_nm,T.oracle); grid on; xlabel('Wavelength (nm)'); ylabel('Normalized spectral power'); legend('true','stale','single pilot','proposed','oracle','Location','best'); exportgraphics(f,fullfile(figDir,'Fig08_reconstruction_example.png'),'Resolution',300); close(f);

% Fig. 9
T=readtable(fullfile(dataDir,'error_vs_temperature.csv')); f=figure('Color','w'); vars={'stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle'}; hold on; for k=1:numel(vars), plot(T.dT_C,T.(vars{k}),'LineWidth',1.2); end; grid on; xlabel('Temperature drift dT (deg C)'); ylabel('Mean relative reconstruction error'); legend(strrep(vars,'_',' '),'Location','best'); exportgraphics(f,fullfile(figDir,'Fig09_error_temperature.png'),'Resolution',300); close(f);

% Fig. 10
T=readtable(fullfile(dataDir,'error_vs_bend.csv')); f=figure('Color','w'); hold on; for k=1:numel(vars), plot(T.curvature_equiv_m_1,T.(vars{k}),'LineWidth',1.2); end; grid on; xlabel('Equivalent bend-curvature perturbation (m^{-1})'); ylabel('Mean relative reconstruction error'); legend(strrep(vars,'_',' '),'Location','best'); exportgraphics(f,fullfile(figDir,'Fig10_error_bend.png'),'Resolution',300); close(f);

% Fig. 11
HP=readmatrix(fullfile(dataDir,'heatmap_proposed.csv')); HS=readmatrix(fullfile(dataDir,'heatmap_stale.csv')); Tg=linspace(-1.5,1.5,13); Bg=linspace(-0.3,0.3,13).*model.b_scale_curv; f=figure('Color','w','Position',[100 100 1000 420]); tiledlayout(f,1,2,'Padding','compact'); nexttile; imagesc(Bg,Tg,HS); axis xy; colorbar; xlabel('Equivalent curvature (m^{-1})'); ylabel('dT (deg C)'); title('Stale calibration'); nexttile; imagesc(Bg,Tg,HP); axis xy; colorbar; xlabel('Equivalent curvature (m^{-1})'); ylabel('dT (deg C)'); title('Dual-pilot Jacobian'); exportgraphics(f,fullfile(figDir,'Fig11_combined_drift_heatmap.png'),'Resolution',300); close(f);

% Fig. 12
T=readtable(fullfile(dataDir,'pilot_separation_conditioning.csv')); f=figure('Color','w'); plot(T.pilot_separation_nm,T.condition_number,'-o','LineWidth',1.3); grid on; xlabel('Pilot wavelength separation (nm)'); ylabel('Jacobian condition number'); exportgraphics(f,fullfile(figDir,'Fig12_pilot_conditioning.png'),'Resolution',300); close(f);

% Fig. 13
T=readtable(fullfile(dataDir,'two_line_resolution.csv')); f=figure('Color','w'); plot(T.line_separation_nm,T.valley_to_peak_ratio,'-o','LineWidth',1.3); hold on; plot(T.line_separation_nm,T.relative_error,'-s','LineWidth',1.3); grid on; xlabel('Two-line separation (nm)'); ylabel('Metric'); legend('Valley/peak','Relative error','Location','best'); exportgraphics(f,fullfile(figDir,'Fig13_resolution.png'),'Resolution',300); close(f);

% Fig. 14
T=readtable(fullfile(dataDir,'jacobian_tolerance.csv')); f=figure('Color','w'); plot(100*T.jacobian_fractional_std,T.mean_reconstruction_error,'-o','LineWidth',1.3); grid on; xlabel('Jacobian calibration error, 1 sigma (%)'); ylabel('Mean relative reconstruction error'); exportgraphics(f,fullfile(figDir,'Fig14_jacobian_tolerance.png'),'Resolution',300); close(f);

% Fig. 15
T=readtable(fullfile(dataDir,'dynamic_tracking.csv')); f=figure('Color','w','Position',[100 100 900 600]); tiledlayout(f,2,1,'Padding','compact'); nexttile; plot(T.frame,T.true_dT_C,'LineWidth',1.2); hold on; plot(T.frame,T.est_dT_C,'LineWidth',1.2); plot(T.frame,T.true_bend_state,'LineWidth',1.2); plot(T.frame,T.est_bend_state,'LineWidth',1.2); grid on; ylabel('State coordinate'); legend('true dT','estimated dT','true bend','estimated bend','Location','best'); nexttile; plot(T.frame,T.stale_error,'LineWidth',1.2); hold on; plot(T.frame,T.proposed_error,'LineWidth',1.2); grid on; xlabel('Frame'); ylabel('Relative reconstruction error'); legend('stale','proposed','Location','best'); exportgraphics(f,fullfile(figDir,'Fig15_dynamic_tracking.png'),'Resolution',300); close(f);
end
