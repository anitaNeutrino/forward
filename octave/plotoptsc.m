% plot the optimal scaling values
function alpha = plotoptsc(z, K, type, p, sigma)
% example
% plot(iwtrans(decfw(wex(3,:)(1:1024),K(1:1024),'d6',4,'wien',[10 400 50 50 50], 4),'d6',4))
% testfwd(3,'d6',4,[10 400 50 50 50],10)
% optimal match at testfwd(3,'d6',4,[110 100 144 432 320],2)
% Strategy: first get the optimal values using rho = 1
% then adjust rho looking at the wavelet transform
% testspike(testy,'d6',4,5,[0.35 0.35 0.3 0.65 0.7],[2 2 1.5 1 1],'soft')
% For anita data: testspike(wex(3,:), 'd10', 5,sqrt(8.5),[0.3 0.3 0.36 0.64 0.49 4],[4 4 4 1 1 1],'hard')

stepsize = 0.1;
errorl = 0.05;
rho = 1;

method = 'hard';

alpha = zeros(1,p+1)+1; % start with scaling value 1

xarray = 0:stepsize:1;
yarray = zeros(p+1,length(xarray));
for i=1:p+1
	%each component of sigma can be between 0 and 1
	j=1;
	for alphaval =xarray
		alpha(i) = alphaval;
		[w, ratiounthr, thrvec] = wienforwd(z(1:1024),K(1:1024),type,p,sigma,alpha, rho,method);
		% update the yarray
		yarray(i,j) = ratiounthr(i); j=j+1;

		% store the minimun difference

		%% No stopping condition. We want the whole graph
		%if (abs(ratiounthr(i) - alphaval) < errorl)
		%	break;
		%end
	end
end

% figure;
% for i=1:p+1
% 	subplot(p+1,1,i)
% 	plot(xarray,yarray(i,:),xarray,xarray);
% end
			

% employ some kind of Newton's method

end

