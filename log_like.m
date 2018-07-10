function C = log_like( calcOut, expectOut)
% Calculates log-likelihood cost
% calcOut = matrix, calculated outputs
% expectOut = matrix, expected outputs for exclusive categories
 
% INTERMEDIATE VARIABLES
% Keep only activations for correct output
catOut = calcOut.*expectOut;   
 
% Cost for non-zero elements
C = sum( -log( catOut(catOut~=0) ) );
 
end
