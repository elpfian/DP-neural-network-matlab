function biasCell = init_biases( nSize )
% Create initial bias matrix for each layer of network and store in cell
% array. Random biases with normal distribution, mean 0, std 1.
% For each bias layer j -> col vector with # row = nodes in j
%
% nSize = vector of # nodes per layer in network
 
% INITIALIZE
% cell 1 not used for calculation, kept for indexing
biasCell = {[]};
 
for i = 2:size( nSize, 2 )
    
    biasCell{i} = randn( nSize(i), 1 );
 
end
 
end
