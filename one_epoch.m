function [wCur, bCur, dwCur, dbCur] =...
    one_epoch( inputs, targets, wCur, bCur, dwCur, dbCur,...
    nodeLayers, batchSize, eta, lambda, mu, transF, costName)
% Creates mini-batches, peforms feedforward, output error,  
% back propagation, and stochastic gradient descent for a single epoch
%
% inputs = matrix, col for each example and row for each input feature
% targets = matrix, col for each example and row for each output feature
% wCur = cell, current weight matrices
% bCur = cell, current bias matrices
% dwCur = cell, change in weights from last update to current
% dbCur = cell, change in biases from last update to current
% nodeLayers = vector, # of nodes in each layer
% batchSize = scalar, # of examples in mini-batch
% eta = scalar, learning rate
% lambda = scalar, coefficient for L2 weight regularization
% mu = scalar, momentum coefficient
% transF = transfer function
% costName = name of cost function
 
%% INTERMEDIATE VARIABLES
trainSize = size( inputs, 2 );
numBatches = ceil( trainSize/batchSize );
depth = size(nodeLayers, 2);
dCur = cell(1, depth);
 
batchStart = 1; % start pointer
 
% MINI-BATCH SHUFFLING
% Index to randomize training set order for each epoch
shuffInd = randperm( trainSize );
 
%% ITERATE OVER BATCHES
for batch = 1:numBatches
    
    % GENERATE MINI-BATCH 
    % In case # batches not even with # inputs
    batchEnd = min( batchStart + batchSize -1, trainSize ); % end pointer
    cBatchSize = ( batchEnd +1) - batchStart; % current batch size
    
    % Select inputs, targets for batch using shuffled indecies
    batchInd = shuffInd( batchStart:batchEnd );
    x_batch = inputs( :, batchInd );    % all rows, shuffled cols
    y_batch = targets( :, batchInd );   % all rows, shuffled cols
        
    % FEEDFORWARD
    [aCur, zCur] = feed_forward( x_batch, wCur, bCur,... 
        depth, transF, costName );
   
    % OUTPUT ERROR, BACKPROPAGATE AND SGD
    [wCur, bCur, dwCur, dbCur] =...
        back_prop_SGD(wCur, bCur, dwCur, dbCur, aCur, zCur, dCur, y_batch,...
        eta, lambda, mu, trainSize, cBatchSize, depth, transF, costName);
    
    batchStart = batchStart + cBatchSize;    % increment batchStart pointer
 
end
 
end
