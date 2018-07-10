function [wCell, bCell, dwCell, dbCell] = ...
    back_prop_SGD(wCell, bCell, dwCell, dbCell,...
    aCell, zCell, dCell, targets, learn, L2reg, mCoef,...
    trnSize, cbSize, depth, tFunc, costName)
% Calculates output error, backpropagates through network 
% and performs SGD. 
% Returns updated weights (wCell) and biases (bCell).
%
% INPUT VARIABLES
% wCell = cell, weights
% bCell = cell, biases
% dwCell = cell, change in weights from last update
% dbCell = cell, change in biases from last update
% dCell = cell, layer errors
% zCell = cell, weighted inputs
% aCell = cell, activations
% targets = matrix, target outputs
% learn = scalar, learning rate
% L2reg = scalar, L2 regularization coefficient
% momentum = scalar, momentum coefficient
% cbSize = scalar, size of current batch
% depth = scalar, # of layers in network
% tFunc = transfer function
 
%% OUTPUT ERROR (of network)
% Varies by cost function
switch costName
    case 'quad_cost'
        dCell{depth} = (aCell{depth} - targets).*tFunc( zCell{depth}, 1 );
 
    case 'log_like' % use only with softmax output layer
        dCell{depth} = (aCell{depth} - targets);
        tFunc = @sigma; % set transfer function to sigma for hidden layers
        
    case 'x_entropy'
        dCell{depth} = (aCell{depth} - targets);
        % NOTE: Output error independent of transfer function
end
 
%% BACKPROPAGATE
for j = (depth-1):-1:2
    dCell{j} = ( wCell{j+1}' * dCell{j+1} ).*tFunc( zCell{j}, 1 );
end
 
%% UPDATE WEIGHTS, BIASES using stochastic gradient descent
% with L2 regularization & momentum
 
% Temporarily store current values of weights & biases
wPrev = wCell;
bPrev = bCell;
 
for j = (depth):-1:2
    
    % Weight =  regularized*Weight + momentum*deltaWeight - eta*nablaC
    wCell{j} = (1 - learn*L2reg/trnSize)*wCell{j}...
        + mCoef*dwCell{j}...
        - (learn/cbSize)*( dCell{j}*aCell{j-1}' );
        
    % Bias = regularized*Bias + momentum*deltaBias - eta*nablaC
    bCell{j} = bCell{j}...
        + mCoef*dbCell{j}...
        - (learn/cbSize)*sum( dCell{j}, 2 );
    
    % Update deltaWeight & deltaBias
    dwCell{j} = wCell{j} - wPrev{j};
    dbCell{j} = bCell{j} - bPrev{j};
end

end
