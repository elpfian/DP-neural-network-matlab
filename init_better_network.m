function [weightCell, biasCell, DwCell, DbCell] =...
    init_better_network( nSize )
% Create initial weight matrix, bias matrix and change in weight and bias
% matrix for each layer of network and store in separate cell arrays. 
% Random weights and biases with normal distribution, mean 0 &
%   weights: std 1/sqrt(# inputs)
%   biases: std 1
% For each weight layer j -> # row = nodes in j, # col = nodes in j-1
% For each bias layer j -> col vector with # row = nodes in j
%
% nSize = vector of # nodes per layer in network
 
% INITIALIZE
depth = size( nSize, 2);
weightCell = cell( 1, depth );
biasCell = cell( 1, depth );
DwCell = cell( 1, depth);
DbCell = cell( 1, depth);
 
% cell 1 not used for calculation, kept for indexing
for i = 2:depth
    
    % Weights   
    weightCell{i} = ( 1/sqrt(nSize(1)) ).*randn( nSize(i), nSize(i-1) );
    DwCell{i} = zeros( size(weightCell{i}) );
    
    % Biases
    biasCell{i} = randn( nSize(i), 1 );
    DbCell{i} = zeros( size(biasCell{i}) );
    
end
 
end
