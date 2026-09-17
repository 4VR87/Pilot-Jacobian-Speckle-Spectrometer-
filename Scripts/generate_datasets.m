function results = generate_datasets(model,dataDir)
%GENERATE_DATASETS Reproduce all manuscript dataset types in MATLAB.
if ~exist(dataDir,'dir'), mkdir(dataDir); end
lam = model.lam(:);
N = model.N; P = model.P;
refIdx = find(abs(lam-model.lam0)==min(abs(lam-model.lam0)),1);

% Spectral correlation
corrL=zeros(N,1);
for k=1:N, corrL(k)=speckle_correlation(model.T0(:,refIdx),model.T0(:,k)); end
writetable(table(lam,corrL,'VariableNames',{'wavelength_nm','corr_to_1550'}),fullfile(dataDir,'spectral_correlation.csv'));

% Thermal/bend correlations
dTs=linspace(-4,4,81).'; corrT=zeros(size(dTs));
for k=1:numel(dTs), corrT(k)=speckle_correlation(model.T0(:,refIdx),field_intensity(model,1550,dTs(k),0)); end
writetable(table(dTs,corrT,'VariableNames',{'dT_C','corr'}),fullfile(dataDir,'thermal_correlation.csv'));
bends=linspace(-0.8,0.8,81).'; corrB=zeros(size(bends));
for k=1:numel(bends), corrB(k)=speckle_correlation(model.T0(:,refIdx),field_intensity(model,1550,0,bends(k))); end
curv=bends.*model.b_scale_curv;
writetable(table(bends,curv,corrB,'VariableNames',{'bend_state','curvature_equiv_m_1','corr'}),fullfile(dataDir,'bend_correlation.csv'));

% Pilot state estimation vs SNR
snrs=(20:5:45).'; tempRMSE=zeros(size(snrs)); bendRMSE=zeros(size(snrs));
for i=1:numel(snrs)
    eT=zeros(50,1); eB=zeros(50,1);
    for r=1:50
        t=-1+2*rand; b=-0.2+0.4*rand;
        q=simulate_state_estimate(model,t,b,snrs(i),0);
        eT(r)=q(1)-t; eB(r)=q(2)-b;
    end
    tempRMSE(i)=sqrt(mean(eT.^2)); bendRMSE(i)=sqrt(mean(eB.^2));
end
writetable(table(snrs,tempRMSE,bendRMSE,'VariableNames',{'pilot_snr_db','temperature_rmse_C','bend_state_rmse'}),fullfile(dataDir,'pilot_state_estimation.csv'));

% Temperature-only bank
bankT=(-1.5:0.25:1.5).'; bank=cell(numel(bankT),1);
for k=1:numel(bankT), bank{k}=build_tmatrix(model,bankT(k),0); end

% Representative reconstruction
sEx=make_spectrum(model,'two',0.20); trueT=1.0; trueB=0.18;
Ttrue=build_tmatrix(model,trueT,trueB); y=add_noise_snr(Ttrue*sEx,35);
q=simulate_state_estimate(model,trueT,trueB,40,0);
That=model.T0+model.JT.*q(1)+model.JB.*q(2);
recStale=reconstruct_spectrum(model,model.T0,y);
recSingle=reconstruct_spectrum(model,single_pilot_update(model,trueT,trueB,40),y);
recProp=reconstruct_spectrum(model,That,y);
recOracle=reconstruct_spectrum(model,Ttrue,y);
writetable(table(lam,sEx,recStale,recSingle,recProp,recOracle, ...
    'VariableNames',{'wavelength_nm','true','stale','single_pilot','proposed','oracle'}), ...
    fullfile(dataDir,'reconstruction_example.csv'));

% Test spectra
specs={make_spectrum(model,'single'),make_spectrum(model,'two',0.20),make_spectrum(model,'broad')};
methods={'stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle'};

% Temperature sweep
Ts=linspace(-1.5,1.5,13).'; E=zeros(numel(Ts),5); bendFixed=0.15;
for it=1:numel(Ts)
    vals=cell(1,5); for k=1:5, vals{k}=[]; end
    for rep=1:3
        for is=1:numel(specs)
            s=specs{is}; Tt=build_tmatrix(model,Ts(it),bendFixed); yy=add_noise_snr(Tt*s,35);
            qq=simulate_state_estimate(model,Ts(it),bendFixed,40,0);
            Tup=model.T0+model.JT.*qq(1)+model.JB.*qq(2);
            [~,ibank]=min(abs(bankT-Ts(it)));
            mats={model.T0,single_pilot_update(model,Ts(it),bendFixed,40),bank{ibank},Tup,Tt};
            for k=1:5, vals{k}(end+1)=relative_error(s,reconstruct_spectrum(model,mats{k},yy)); end %#ok<AGROW>
        end
    end
    for k=1:5, E(it,k)=mean(vals{k}); end
end
Ttab=array2table([Ts,E],'VariableNames',[{'dT_C'},methods]); writetable(Ttab,fullfile(dataDir,'error_vs_temperature.csv'));

% Bend sweep
Bg=linspace(-0.30,0.30,13).'; E=zeros(numel(Bg),5); tempFixed=0.8;
for ib=1:numel(Bg)
    vals=cell(1,5); for k=1:5, vals{k}=[]; end
    for rep=1:3
        for is=1:numel(specs)
            s=specs{is}; Tt=build_tmatrix(model,tempFixed,Bg(ib)); yy=add_noise_snr(Tt*s,35);
            qq=simulate_state_estimate(model,tempFixed,Bg(ib),40,0);
            Tup=model.T0+model.JT.*qq(1)+model.JB.*qq(2);
            [~,ibank]=min(abs(bankT-tempFixed));
            mats={model.T0,single_pilot_update(model,tempFixed,Bg(ib),40),bank{ibank},Tup,Tt};
            for k=1:5, vals{k}(end+1)=relative_error(s,reconstruct_spectrum(model,mats{k},yy)); end %#ok<AGROW>
        end
    end
    for k=1:5, E(ib,k)=mean(vals{k}); end
end
curv=Bg.*model.b_scale_curv;
Btab=array2table([Bg,curv,E],'VariableNames',[{'bend_state','curvature_equiv_m_1'},methods]); writetable(Btab,fullfile(dataDir,'error_vs_bend.csv'));

% 2-D error grids
Tgrid=linspace(-1.5,1.5,13); Bgrid=linspace(-0.3,0.3,13);
heatProp=zeros(numel(Tgrid),numel(Bgrid)); heatStale=heatProp; s=make_spectrum(model,'two',0.20);
for it=1:numel(Tgrid)
    for ib=1:numel(Bgrid)
        ep=zeros(2,1); es=zeros(2,1);
        for rep=1:2
            Tt=build_tmatrix(model,Tgrid(it),Bgrid(ib)); yy=add_noise_snr(Tt*s,35);
            qq=simulate_state_estimate(model,Tgrid(it),Bgrid(ib),40,0);
            Tup=model.T0+model.JT.*qq(1)+model.JB.*qq(2);
            ep(rep)=relative_error(s,reconstruct_spectrum(model,Tup,yy));
            es(rep)=relative_error(s,reconstruct_spectrum(model,model.T0,yy));
        end
        heatProp(it,ib)=mean(ep); heatStale(it,ib)=mean(es);
    end
end
writematrix(heatProp,fullfile(dataDir,'heatmap_proposed.csv')); writematrix(heatStale,fullfile(dataDir,'heatmap_stale.csv'));

% Pilot separation conditioning
seps=(1.0:0.5:7.0).'; cn=zeros(size(seps)); nT=zeros(size(seps)); nB=zeros(size(seps));
for k=1:numel(seps)
    [~,i1]=min(abs(lam-(1550-seps(k)/2))); [~,i2]=min(abs(lam-(1550+seps(k)/2)));
    Gs=[vertcat(model.JT(:,i1),model.JT(:,i2)),vertcat(model.JB(:,i1),model.JB(:,i2))];
    cn(k)=cond(Gs); nT(k)=norm(Gs(:,1)); nB(k)=norm(Gs(:,2));
end
writetable(table(seps,cn,nT,nB,'VariableNames',{'pilot_separation_nm','condition_number','thermal_jacobian_norm','bend_jacobian_norm'}),fullfile(dataDir,'pilot_separation_conditioning.csv'));

% Two-line resolution
seps2=(0.05:0.025:0.50).'; valley=zeros(size(seps2)); rerr=zeros(size(seps2));
for k=1:numel(seps2)
    s=make_spectrum(model,'two',seps2(k),1550,0.07); Tt=build_tmatrix(model,0.8,0.15); yy=add_noise_snr(Tt*s,38);
    qq=simulate_state_estimate(model,0.8,0.15,42,0); Tup=model.T0+model.JT.*qq(1)+model.JB.*qq(2);
    r=reconstruct_spectrum(model,Tup,yy); [~,i0]=min(abs(lam-1550));
    lp=max(r(lam<1550)); rp=max(r(lam>1550)); valley(k)=r(i0)/max(min(lp,rp),1e-12); rerr(k)=relative_error(s,r);
end
writetable(table(seps2,valley,rerr,'VariableNames',{'line_separation_nm','valley_to_peak_ratio','relative_error'}),fullfile(dataDir,'two_line_resolution.csv'));

% Jacobian tolerance
jerr=[0,0.0025,0.005,0.01,0.02,0.03,0.05].'; meanRec=zeros(size(jerr)); meanState=zeros(size(jerr));
for k=1:numel(jerr)
    er=zeros(15,1); se=zeros(15,1);
    for rep=1:15
        t=-1+2*rand; b=-0.2+0.4*rand; s=make_spectrum(model,'two',0.2); Tt=build_tmatrix(model,t,b); yy=add_noise_snr(Tt*s,35);
        qq=simulate_state_estimate(model,t,b,40,jerr(k));
        JTu=model.JT.*(1+jerr(k).*randn(size(model.JT))); JBu=model.JB.*(1+jerr(k).*randn(size(model.JB)));
        r=reconstruct_spectrum(model,model.T0+JTu.*qq(1)+JBu.*qq(2),yy);
        er(rep)=relative_error(s,r); se(rep)=abs(qq(1)-t)+abs(qq(2)-b);
    end
    meanRec(k)=mean(er); meanState(k)=mean(se);
end
writetable(table(jerr,meanRec,meanState,'VariableNames',{'jacobian_fractional_std','mean_reconstruction_error','mean_abs_state_error_sum'}),fullfile(dataDir,'jacobian_tolerance.csv'));

% Dynamic tracking
frames=(0:119).'; trueTemp=0.9*sin(2*pi*frames/80)+0.25*sin(2*pi*frames/23);
trueB=0.16*sin(2*pi*frames/53+0.7)+0.05*cos(2*pi*frames/17);
estTemp=zeros(size(frames)); estB=zeros(size(frames)); staleErr=zeros(size(frames)); propErr=zeros(size(frames)); sDyn=make_spectrum(model,'two',0.20);
for k=1:numel(frames)
    qq=simulate_state_estimate(model,trueTemp(k),trueB(k),40,0); estTemp(k)=qq(1); estB(k)=qq(2);
    Tt=build_tmatrix(model,trueTemp(k),trueB(k)); yy=add_noise_snr(Tt*sDyn,35);
    staleErr(k)=relative_error(sDyn,reconstruct_spectrum(model,model.T0,yy));
    propErr(k)=relative_error(sDyn,reconstruct_spectrum(model,model.T0+model.JT.*qq(1)+model.JB.*qq(2),yy));
end
writetable(table(frames,trueTemp,estTemp,trueB,estB,staleErr,propErr, ...
    'VariableNames',{'frame','true_dT_C','est_dT_C','true_bend_state','est_bend_state','stale_error','proposed_error'}),fullfile(dataDir,'dynamic_tracking.csv'));

% Update burden table
Method={'Full wavelength recalibration';'Temperature-bank lookup';'Single-pilot correction';'Proposed dual-pilot Jacobian'};
reference_frames_per_update=[81;0;1;2]; matrix_updates=[81;1;1;2]; relative_update_time_vs_fullscan=[1.0;0.0062;0.0062;0.0124]; latent_parameters_tracked=[0;0;1;2];
writetable(table(Method,reference_frames_per_update,matrix_updates,relative_update_time_vs_fullscan,latent_parameters_tracked),fullfile(dataDir,'update_cost.csv'));

% Summary
results=struct(); results.pilot_jacobian_condition_number=cond(model.G); results.example_stale_relerr=relative_error(sEx,recStale); results.example_single_relerr=relative_error(sEx,recSingle); results.example_proposed_relerr=relative_error(sEx,recProp); results.example_oracle_relerr=relative_error(sEx,recOracle); results.dynamic_mean_stale_error=mean(staleErr); results.dynamic_mean_proposed_error=mean(propErr);
fid=fopen(fullfile(dataDir,'summary.txt'),'w');
fprintf(fid,'pilot_jacobian_condition_number=%.15g\n',results.pilot_jacobian_condition_number);
fprintf(fid,'example_true_dT_C=1.0\nexample_true_bend_state=0.18\n');
fprintf(fid,'example_stale_relerr=%.15g\n',results.example_stale_relerr);
fprintf(fid,'example_single_relerr=%.15g\n',results.example_single_relerr);
fprintf(fid,'example_proposed_relerr=%.15g\n',results.example_proposed_relerr);
fprintf(fid,'example_oracle_relerr=%.15g\n',results.example_oracle_relerr);
fprintf(fid,'dynamic_mean_stale_error=%.15g\n',results.dynamic_mean_stale_error);
fprintf(fid,'dynamic_mean_proposed_error=%.15g\n',results.dynamic_mean_proposed_error); fclose(fid);
end
