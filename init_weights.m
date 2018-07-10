function weightCell = init_weights( nSize )
% Create initial weight matrix for each layer of network and store in cell
% array. Random weights with normal distribution, mean 0, std 1.
% For each weight layer j -> # row = nodes in j, # col = nodes in j-1
%
% nSize = vector of # nodes per layer in network
 
% INITIALIZE
% cell 1 not used for calculation, kept for indexing
weightCell = {[]};
 
for i = 2:size( nSize, 2 )
    
    sizeCurr = nSize(i);
    sizePrev = nSize(i-1);
    
    weightCell{i} = randn( sizeCurr, sizePrev );
 
end
 
end
