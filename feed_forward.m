function [aCell, zCell] = feed_forward( xMat, wCell, bCell,...
    depth, tFunc, cName )
% Propagates input xMat through network defined by wCell and bCell with 
% transfer function tFunc.
% Returns cell array of activations (aCell) and weighted inputs (zCell).
%
% FUNCTION VARIABLES
% xMat = matrix, inputs
% wCell = cell array of weights
% bCell = cell array of biases
% depth = scalar, # of layers in network
% tFunc = transfer function
% cName = string, name of cost function
 
% INITIALIZATIONS
aCell = cell(1, depth);
aCell{1} = xMat;
zCell = cell(1, depth);
 
%% VARY BY TRANSFER/COST FUNCTION
% Softmax (tFunc) - use only for output layer, use sigma for hidden layers
% Cross-entropy (cName) - use sigma for output layer, use tFunc for hidden layers
 
if strcmp(func2str(tFunc), 'softmax')   % SOFTMAX
    % Hidden layers use sigma
    netTFunc = @sigma;    
    for j = 2:(depth-1)
        zCell{j} = wCell{j}*aCell{j-1} + repmat( bCell{j}, 1, size(xMat,2) );
        aCell{j} = netTFunc( zCell{j}, 0 );
    end
    
    % Output layer uses softmax
    zCell{depth} = wCell{depth}*aCell{depth-1}...
        + repmat( bCell{depth}, 1, size(xMat,2) );
    aCell{depth} = tFunc( zCell{depth}, 0 );
   
elseif strcmp(cName, 'x_entropy')       % CROSS-ENTROPY
    % Hidden layers use tFunc
    for j = 2:(depth-1)
        zCell{j} = wCell{j}*aCell{j-1} + repmat( bCell{j}, 1, size(xMat,2) );
        aCell{j} = tFunc( zCell{j}, 0 );
    end
    
    % Output layer uses sigma
    outFunc = @sigma;
    zCell{depth} = wCell{depth}*aCell{depth-1}...
        + repmat( bCell{depth}, 1, size(xMat,2) );
    aCell{depth} = outFunc( zCell{depth}, 0 );
    
else                            % TANH, SIGMA, RELU (w/o CROSS-ENTROPY)
    for j = 2:depth
        zCell{j} = wCell{j}*aCell{j-1} + repmat( bCell{j}, 1, size(xMat,2) );
        aCell{j} = tFunc( zCell{j}, 0 );
    end
end
 
end
