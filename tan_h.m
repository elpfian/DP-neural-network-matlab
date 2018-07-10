function a = tan_h( X, dFlag )
% Calculates hyperbolic tangent or derivative
% X = matrix
% dFlag = 0 indicates function, 1 indicates take derivative
 
% tanh
a = tanh(X);
 
% tanh derivative
if dFlag
    a = 1-tanh(X).^2;  
end
 
end
