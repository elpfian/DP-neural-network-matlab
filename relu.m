function a = relu( X, dFlag )
% Calculates rectified linear unit or derivative
% X = matrix
% dFlag = 0 indicates function, 1 indicates take derivative
 
% ReLU
a = max(0, X);
 
% ReLU derivative
if dFlag
    a(a>0) = 1;  
end
 
end
