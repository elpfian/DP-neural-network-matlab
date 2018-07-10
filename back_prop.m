function  [wNew, bNew] = back_prop(xMat, yMat,...
    wCell, bCell, depth, eta, n)
% Completes feedforward and backpropagation for a neural network, then
% updates weights and biases. Returns 2 cell arrays: wNew and bNew.
%
% xMat = matrix, inputs: row = features, col = example
% yMat = matrix, targets: row = features, col = example
% wCell = cell array, initial weight matrices
% bCell = cell array, initial bias (col) vectors
% depth = scalar, # of layers in network
% eta = scalar, learning rate
% n = scalar, optional batch size. If omitted, online update.
%
% REQUIRED FUNCTION FILES: sigma, sigma_prime
 
% CHECK OPTIONAL PARAMETER batch size
% If missing, set n = 1 (online update)
if (~exist( 'n', 'var' ))
    n = 1;
end
 
% INITIALIZE activations, weighted inputs and deltas
% zCell{1} and dCell{1} are not used in calculation but kept for indexing
aCell = { sigma(xMat) };    % THIS IS WRONG - should not apply activation function here, aCell = { xMat };
zCell = { [] };
dCell = { [] };
 
% FEEDFORWARD
for i = 2:depth      % iterate over depth of network
    
    zCell{i} = wCell{i}*aCell{i-1} + repmat( bCell{i}, 1, size(xMat, 2));
    aCell{i} = sigma( zCell{i} );
    
end
 
% OUTPUT ERROR (of network)
% quad_nabla function defined below
dCell{depth} = ( aCell{depth} - yMat ).*sigma_prime( zCell{depth} );
 
% BACKPROPAGATE ERROR
for i = ( depth-1 ):-1:2  % iterate back through network
    
    dCell{i} = ( wCell{i+1}' * dCell{i+1} ).*sigma_prime( zCell{i} );
 
end
 
% UPDATE WEIGHTS, BIASES
% Iterate backward through network
% Sum for cases in mini-batch
for i = ( depth ):-1:2
    
    % dot-product computes sum
    wNew{i} = wCell{i} - (eta/n)*( dCell{i}*aCell{i-1}' ); 
    
    bNew{i} = bCell{i} - (eta/n)*sum( dCell{i}, 2 );    % sum rows
    
end
 
end         % END back_prop
