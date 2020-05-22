
function [Vout] = waveletDecon(Vin, Psf, noise_sd, alpha, rho)

% Based on script written by Debdeep Bhattacharya -- P. Gorham April 2020
%noise_sd: standard deviation of noise present in the observed signal
% Need to compute this separately
%  wavelet deconvolution using ForWard method

% alpha: Fourier shrinkage parameter, ideally it needs to be computed a 
% posteriori using bisection method to minimize error, use getoptsc() (caution: slow)
% It is a vector of length (p+1), alpha(j) is between 0 and 1 for each j
% higher alpha_j implies more signal has been damaged by noise

% using p=5 for this version

% Vector of length (p+1), gives extra handle on thesholding noise at 
% different resolution level
% default value is 1 for all rho_j
% higher value of rho_j means more aggressive thresholding at level j
% How to use: look at the multi-resolution plot of the wavelet transform 
% using plotcoeffs() or plotthr() to identify the level at which an 
% artifacts appears, increase rho for that level to remove it



% load the observed signal and the impulse response
% make sure both are of same length, a power of 2

n1 = length(Vin);
n2 = length(Psf);
if n1 ~= n2
    error('Vin and Psf must be same length');
    return
end

if(~isrow(Vin))
    Vin = Vin';
end
if(~isrow(Psf))
    Psf = Psf';
end

% size(Vin)
% size(Psf)

p2 = nextpow2(n1);
z = Vin
K = Psf
% z = padarray(Vin, [0 2^p2-n1],0,'post');
% K = padarray(Psf, [0 2^p2-n2],0,'post');

% size(z);
% size(K);
% pause;

% load the observed signal
%z = load('sample_z.txt');


% load the impulse response 
%K = load('sample_K.txt');

%figure, plot(z), hold on, plot(K), hold off;

% make sure they are row vectors
%z = z';
%K = K';
% Check that the vectors are of the same length and a power of 2
size(z) == size(K);
log2(length(z)) == floor(log2(length(z)));

%% set up the parameters

% filter type
% choose a type defined in filt.m
type = 'meyer'; 	% you can try 'd10' with different alpha, rho	

% resolution of the wavelet decomposition
% the higher this value is, the more number of handles we have in terms of thresholding the coarser levels
p = 5;						

% standard deviation of noise present in the observed signal
% Need to compute this separately, or we can feed a noise vector of size(z)
%noise_sd = 11.5;	% computed for WAIS signals

% Fourier shrinkage parameter, ideally it needs to be computed a posteriori using bisection method to minimize error, use getoptsc() (caution: slow)
% It is a vector of length (p+1), alpha(j) is between 0 and 1 for each j
% higher alpha_j implies more signal has been damaged by noise
% alpha = [0.5 0.5 0.5 0.5 0.5 0.5];	% sample value
% alpha = [0.1 0.1 0.1 0.1 0.1 0.1];	% sample value

% Vector of length (p+1), gives extra handle on thesholding noise at different resolution level
% default value is 1 for all rho_j
% higher value of rho_j means more aggressive thresholding at level j
% How to use: look at the multi-resolution plot of the wavelet transform using plotcoeffs() or plotthr() to identify the level at which an artifacts appears, increase rho for that level to remove it
% rho = [2 3 4.5 3 2 10e10];	% worked well for WAIS signals
% rho = [1 1 1 1 1 1];

% rule of thresholding
rule = 'soft';	% better than hard thesholding

%% compute the wavelet transform w of the deconvolved signal
% thrvec is a vector of values at which different wavelet coefficients were thsholded
[w thrvec] = wienforwd(z, K, type, p, noise_sd, alpha, rho, rule);

%% look at the multiresolution plot of the w
%plotthr(w, p, thrvec);

%% compute the inverse wavelet transform of w
output = iwtrans(w, type, p);

%% Plot the output
%figure, plot(output);

Vout = output;

return

end
