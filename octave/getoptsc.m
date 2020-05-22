% get the optimal scaling values
function alpha = getoptsc(z, K, type, p, sigma, rootmethod)
% example
% plot(iwtrans(decfw(wex(3,:)(1:1024),K(1:1024),'d6',4,'wien',[10 400 50 50 50], 4),'d6',4))
% testfwd(3,'d6',4,[10 400 50 50 50],10)
% optimal match at testfwd(3,'d6',4,[110 100 144 432 320],2)
% Strategy: first get the optimal values using rho = 1
% then adjust rho looking at the wavelet transform
% testspike(testy,'d6',4,5,[0.35 0.35 0.3 0.65 0.7],[2 2 1.5 1 1],'soft')
% For anita data: testspike(wex(3,:), 'd10', 5,sqrt(8.5),[0.3 0.3 0.36 0.64 0.49 4],[4 4 4 1 1 1],'hard')

stepsize = 0.02;
errorl = 0.04;

rho = 1;
method = 'hard';

alpha = zeros(1,p+1)+1; % start with scaling value 1

z = z(1:1024);
K = K(1:1024);

% use bisection method


switch rootmethod
case 'bisec'
for i=1:p+1
	% initial guess
	a = 0.1;
	b = 0.9;
	while (abs(a-b)>errorl)
	alpha(i) = a;
	[w, ratiounthr, thrvec] = wienforwd(z,K,type,p,sigma,alpha, rho,method);
	fa = ratiounthr(i);

	alpha(i) = b;
	[w, ratiounthr, thrvec] = wienforwd(z,K,type,p,sigma,alpha, rho,method);
	fb = ratiounthr(i);

	
	if ((a-fa)>=0 && (b - fb)<=0)
		c = (a+b)/2;
		alpha(i) = c;
		[w, ratiounthr, thrvec] = wienforwd(z,K,type,p,sigma,alpha, rho,method);
		fc = ratiounthr(i);
		if ((c -fc) >0)
			a = c;
		else
			b = c;
		end
	elseif ((a-fa)<=0 && (b - fb)>=0)
		c = (a+b)/2;
		alpha(i) = c;
		[w, ratiounthr, thrvec] = wienforwd(z,K,type,p,sigma,alpha, rho,method);
		fc = ratiounthr(i);
		if ((c -fc) >0)
			b = c;
		else
			a = c;
		end
	else
		i
		c = 0.1;
		disp('No solution in between');
		break;
	end
	end
	alpha(i) = c;
end



case 'search'


xarray = 0.1:stepsize:1;
for i=1:p+1
	%each component of sigma can be between 0 and 1
	for alphaval =xarray
		alpha(i) = alphaval;
		[w, ratiounthr, thrvec] = wienforwd(z(1:1024),K(1:1024),type,p,sigma,alpha, rho,method);
		if (abs(ratiounthr(i) - alphaval) < errorl)
			break;
		end
	end
end

otherwise
	disp('wrong rootfinding method')
end


			

