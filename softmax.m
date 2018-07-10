function a = softmax( X, ~ )
% Calculates softmax of final outputs or derivative
% X = matrix
% dFlag = 0 indicates function, 1 indicates take derivative
 
% INTERMEDIATE VARIABLES
b = size(X, 1);     % # of input rows
 
a = exp(X)./repmat( sum(exp(X)), b, 1);
 
% Softmax derivative
%if dFlag
%    ???? Not needed for this project!  
%end
 
end
