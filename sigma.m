function a = sigma( X, dFlag )
% Calculates logsig or derivative
% X = matrix
% dFlag = 0 indicates function, 1 indicates take derivative
 
% Sigma
a = 1./(1 + exp(-X) );
 
% Sigma derivative
if dFlag
    a = a.*( 1-a );  
end
 
end
