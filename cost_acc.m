function [correct, cost, accuracy] = ...
    cost_acc( output, targets, samples, costF, L2Reg )
% Calculates # correct, cost and accuracy
% output = matrix, calculated output
% targets = matrix, target output
% samples = scalar, # of samples in outputs/targets
% costF = cost function
% L2Reg = scalar, weight minimization term
 
% Correct
if max(output(:)) < .5  % Zero outputs >= .5, use max
    correct = sum( all( targets == max(output, 2), 1 ), 2 );
else    % One output >= .5, use round
    correct = sum( all( targets == round(output), 1 ), 2 );
end
 
% Cost + weight minimization term
cost = costF(output, targets) + L2Reg;
 
% Accuracy
accuracy = correct / samples ;
 
end
