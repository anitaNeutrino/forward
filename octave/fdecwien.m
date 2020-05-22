% Wiener deconvolution in the frequency domain
function [fw, mult] = fdecwien(fsig, fimp, noise, scaling)


N = length(fsig);
if ~exist('scaling')	% scaling is optional
	scaling = 1;
end

% assume that the fsig and fimp are of the same length

% naive deconvolution in the fourier domain
% skip to avoid dividing by zero
%wfft = fsig./fimp;

	% definition
	hsq = abs(fimp).^2;

	% model of the original signal
	%fori = fsig;
	fori = fsig;

	% construct the multiplier
	if (length(noise)==1)	% noise is actually the variance
		sigmasq = N*noise^2;
	else			% it is the raw noise
		sigmasq = abs(fft(noise)).^2;
	end
	%mult = hsq ./( hsq + scaling*N*(sigma.^2)./(abs(fori).^2));
	%mult = hsq ./( hsq + scaling*(sigmasq)./(abs(fori).^2));
	%mult = hsq ./( hsq + scaling*(sigmasq)./((abs(fori).^2) .*(hsq) ));

	%fw = fsig .* conj(fimp) ./( hsq + scaling*(sigmasq)./(abs(fori).^2));
	% to correct the error of considering y as x, we multiply by the norm of h  so that ||xtil|| = ||x|| 
	% where xtilhat = yhat/hhat*||hhat||, hat is Fourier transform
	% This gives a much better result while doing wiener
	fw = fsig .* conj(fimp) ./( hsq + scaling*(sigmasq)./((abs(fori).^2)./var(fimp)) );
	% we need to output multi as well, for wienforwd.m
	mult = hsq ./( hsq + scaling*(sigmasq)./(abs(fori).^2).*(var(fimp)) );
% multiply by the scaling and take inverse fourier transform
%fw = wfft.*mult;
