function a = sigma_prime( X )
% Calculates logsig derivative
% X = column vector
%
% REQUIRED FUNCTION FILES: sigma, sigma_prime
 
a = sigma(X).*( 1-sigma(X) );
 
end
