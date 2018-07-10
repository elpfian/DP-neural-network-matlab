function C = x_entropy( calcOut, expOut)
% Calculates cross entropy cost
% calcOut = matrix, calculated outputs
% expectOut = matrix, expected outputs 
 
% INTERMEDIATE VARIABLES
samples = size( calcOut, 2 );
 
% Cost
C = -(1/samples).*sum( sum( expOut.*log(calcOut) + (1-expOut).*log(1-calcOut) ) );
 
end
