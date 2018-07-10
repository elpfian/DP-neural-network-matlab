function C = meanSqErr( calcOut, expectOut)
% Calculates mean squared error.
% calcOut = matrix, calculated outputs
% expectOut = matrix, expected outputs
 
% INTERMEDIATE VARIABLES
samples = size( calcOut, 2 );
diff = expectOut - calcOut;
 
% INITIALIZE cost
cost = zeros(1, samples);
 
% Apply norm to difference for each sample and square
for i = 1:samples
    
    cost(i) = ( norm( diff(i) ) )^2;
    
end
 
C = ( 1/samples ) * sum( cost ); 
 
end
