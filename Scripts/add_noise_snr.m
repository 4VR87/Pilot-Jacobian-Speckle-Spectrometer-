function y = add_noise_snr(v,snr_db)
%ADD_NOISE_SNR Add white Gaussian noise using amplitude SNR convention.
sig = sqrt(mean(v(:).^2));
sigma = sig/(10^(snr_db/20));
y = v + sigma.*randn(size(v));
end
