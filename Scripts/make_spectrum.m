function s = make_spectrum(model,kind,varargin)
%MAKE_SPECTRUM Generate manuscript test spectra.
lam = model.lam(:);
switch lower(kind)
    case 'single'
        center = 1550.0; fwhm = 0.12;
        if numel(varargin)>=1, center=varargin{1}; end
        if numel(varargin)>=2, fwhm=varargin{2}; end
        s = exp(-4*log(2)*(lam-center).^2/fwhm^2);
    case 'two'
        sep=0.20; center=1550.0; fwhm=0.09;
        if numel(varargin)>=1, sep=varargin{1}; end
        if numel(varargin)>=2, center=varargin{2}; end
        if numel(varargin)>=3, fwhm=varargin{3}; end
        s = 0.9*exp(-4*log(2)*(lam-(center-sep/2)).^2/fwhm^2) + ...
            exp(-4*log(2)*(lam-(center+sep/2)).^2/fwhm^2);
    case 'broad'
        center=1550.2; fwhm=1.2;
        if numel(varargin)>=1, center=varargin{1}; end
        if numel(varargin)>=2, fwhm=varargin{2}; end
        s = exp(-4*log(2)*(lam-center).^2/fwhm^2) .* ...
            (1+0.12*cos(2*pi*(lam-center)/0.4));
        s = max(s,0);
    otherwise
        error('Unknown spectrum type: %s',kind);
end
s = s./max(s);
end
