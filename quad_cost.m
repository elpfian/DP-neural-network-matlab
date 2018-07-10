function C = quad_cost( calcOut, expectOut)
% Calculates quadratic cost
% calcOut = matrix, calculated outputs
% expectOut = matrix, expected outputs
 
% INTERMEDIATE VARIABLES
samples = size( calcOut, 2 );
 
% Quadratic cost
C = sum( sum( (expectOut-calcOut).^2, 2) )/(2*samples);
 
end
