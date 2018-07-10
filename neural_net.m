function [wCur, bCur, acc, cost] = neural_net( inputs, targets,...
    nodeLayers, numEpochs, batchSize, split, eta, mu, lambda,...
    transF, costF, wStart, bStart)
% Learns a neural network. Returns learned weights(wCur) and biases(bCur)
% and accuracy(acc) & cost(cost) per epoch for training, 
% test and validation. Prints results for each epoch.
%
%% FUNCTION DATA
% INPUT VARIABLES
% inputs = matrix, col for each example and row for each input feature
% targets = matrix, col for each example and row for each output feature
% nodeLayers = vector, # of nodes in each layer
% numEpochs = scalar, # of epochs to run
% batchSize = scalar, # of examples in mini-batch
% split = vector, proportion of examples for train, test and validation
% eta = scalar, learning rate
% mu = scalar, momentum coefficient
% lambda = scalar, coefficient for L2 weight regularization
% transF = transfer function
% costF = cost function
% wStart (optional) = cell, initial weight matrices
% bStart (optional) = cell, initial bias matrices
%
% OUTPUT VARIABLES
% wCur = cell, learned weight matrices for each layer of network
% bCur = cell, learned biase matrices for each layer of network
% acc = matrix, col for accuracy per epoch for rows train, test & validation
% cost = matrix, col for cost per epoch for rows train, test & validation
%
% REQUIRED FUNCTION FILES: init_better_network, feed_forward,
% one_epoch (REQUIRES back_prop_SGD), cost_acc, screen_print, 
% COST FUNCTIONS(log_like, quad_cost, x_entropy), 
% TRANSFER FUNCTIONS(relu, sigma, softmax, tan_h)
 
%% INPUT CHECKS
transName = func2str(transF);
costName = func2str(costF);
 
% Valid number of cases in inputs vs targets
if size( inputs, 2 ) ~= size( targets, 2 )
    error('Input and targets must have same number of cases.');
    return                              % EXIT NEURAL_NET
% Valid network size for inputs, targets
elseif size( inputs, 1) ~= nodeLayers(1)...
        || size( targets, 1 ) ~= nodeLayers(end);   
    error('Incorrect network size.');
    return                              % EXIT NEURAL_NET
% Valid transfer function
elseif isempty( strfind( 'sigma,relu,tan_h,softmax', transName ) )
    error('Invalid transfer function.');
    return                              % EXIT NEURAL_NET
% Valid cost function
elseif isempty( strfind( 'log_like,quad_cost,x_entropy', costName ) )
    error('Invalid cost function.');
    return                              % EXIT NEURAL_NET
% All positive parameters
elseif any([numEpochs, batchSize, eta, lambda, mu] < 0 )
    error('Parameters cannot be negative.');
    return                              % EXIT NEURAL_NET
% If supplied starting weights or biases, supplied both    
elseif xor( exist('wStart', 'var'), exist('bStart', 'var') )
    error('If supplying starting weights and biases, must supply both.');
    return                              % EXIT NEURAL_NET
end
 
%% INITIALIZATIONS
% Accuracy & cost
acc = zeros(3, numEpochs);  % rows for train, test, validation
cost = zeros(3, numEpochs); % rows for train, test, validation
 
% Stopping conditions
accTrn = 0;      % initialize training accuracy
stop = 10;       % stop if no improvement in this # of epochs
countEpoch = 1;  % epoch counter
 
% Intermediate variables
netDepth = size(nodeLayers, 2);
numCases = size( inputs, 2 );
 
% Weights, biases, and change in weights/biases
if ~exist('wStart', 'var') && ~exist('bStart', 'var') 
    % No weights/biases supplied
    [wCur, bCur, dwCur, dbCur] = init_better_network( nodeLayers );
    
else % Supplied weights/biases
    wCur = wStart;
    bCur = bStart;
    
    % Initialize change in weight/bias
    dwCur = cell( 1, netDepth);
    dbCur = cell( 1, netDepth);
 
    % cell 1 not used for calculation, kept for indexing
    for i = 2:netDepth
        dwCur{i} = zeros( size(wCur{i}) );
        dbCur{i} = zeros( size(bCur{i}) );
    end
end     % End weights, biases, etc.
 
%% GENERATE TRAIN/TEST/VALIDATION SETS
% Create randomized index
splitInd = randperm( numCases ); 
 
% Use "split" variable to get size of each subset
numVal = round( split(3)*numCases );
numTst = round( split(2)*numCases );
numTrn = numCases - numTst - numVal; % In case of uneven split
 
% Assign inputs using randomized index
inputTrn = inputs( :, splitInd(1:numTrn) );
inputTst = inputs( :, splitInd( (numTrn+1):(numTrn+numTst) ) );
inputVal = inputs( :, splitInd( (numTrn+numTst+1):end ) );
 
% Assign targets using randomized index
targetTrn = targets( :, splitInd(1:numTrn) );
targetTst = targets( :, splitInd( (numTrn+1):(numTrn+numTst) ) );
targetVal = targets( :, splitInd( (numTrn+numTst+1):end ) );
 
%% MAIN LOOP
% Print parameters for reference
fprintf('\nEps:%d, Batch:%d, eta:%.2f, ', numEpochs, batchSize, eta)
fprintf('Trans: %s, Cost: %s, Mom:%.1f, Reg:%.1f\n',...
    transName, costName, mu, lambda);
                               
% Stop when numEpochs completed or training accuracy = 100%
while (countEpoch <= numEpochs) && (accTrn < 1);    % START WHILE loop
        
    %% LEARN NETWORK for one epoch using training data
    [wCur, bCur, dwCur, dbCur] =...
    one_epoch( inputTrn, targetTrn, wCur, bCur, dwCur, dbCur,...
    nodeLayers, batchSize, eta, lambda, mu, transF, costName);
    
    %% GENERATE NETWORK OUTPUT, COST, ACCURACY for entire epoch on each set
    
    % CALCULATE weight minimization term for L2 Regularization
    % sum of (all weights squared)
    totWSq = 0;
    for i = 2:netDepth     % wCur{1} is empty
        totWSq = totWSq + sum( sum(wCur{i}.^2) );
    end
    regTerm = (lambda/2/numCases)*totWSq;
 
    % TRAIN
    % Output
    [active, ~ ] =...
        feed_forward( inputTrn, wCur, bCur, netDepth, transF, costName );
    output = active{netDepth};
    


    % Correct, cost, accuracy
    [corTrn, costTrn, accTrn] = ...
        cost_acc( output, targetTrn, numTrn, costF, regTerm );
 
    % TEST
    if numTst == 0  % If no test set from split
        corTst = 0;
        costTst = 0;
        accTst = 0;
    else            % Test set designated
        % Output
        [active, ~ ] =...
            feed_forward( inputTst, wCur, bCur, netDepth, transF, costName );
        output = active{netDepth};
        
        % Correct, cost, accuracy
        [corTst, costTst, accTst] = ...
            cost_acc( output, targetTst, numTst, costF, regTerm );
    end             % End TEST
    
    % VALIDATION
    if numVal == 0      % If no validation set from split
        corVal = 0;
        costVal = 0;
        accVal = 0;
    else                % Validation set designated
        % Output
        [active, ~ ] =...
            feed_forward( inputVal, wCur, bCur, netDepth, transF, costName );
        output = active{netDepth};
        
        % Correct, cost, accuracy
        [corVal, costVal, accVal] = ...
            cost_acc( output, targetVal, numVal, costF, regTerm );
 
        % Update best validation cost for stopping condition
        % 1st time through or current cost less than previous min
        if  ~exist('costValMin', 'var') || (costVal < costValMin)
            costValMin = costVal;
        end
 
    end                 % End VALIDATION
    
    % PRINT
    if numEpochs < 100
        screen_print(countEpoch, costTrn, corTrn, numTrn, accTrn,...
            costTst, corTst, numTst, accTst, costVal, corVal, numVal, accVal)
        
    elseif (numEpochs > 100) &&...      % Print every 10 & last epoch
            ( (mod(countEpoch,10) == 0) || (countEpoch == numEpochs) )
        screen_print(countEpoch, costTrn, corTrn, numTrn, accTrn,...
            costTst, corTst, numTst, accTst, costVal, corVal, numVal, accVal)
    end
    
    % UPDATE ACCURACY AND COST matrices
    acc(:, countEpoch) = [accTrn; accTst; accVal];
    cost(:, countEpoch) = [costTrn; costTst; costVal];
    
    %% EARLY STOPPING
    % At least stop # of epochs has passed and minimum validation cost
    % has not occurred in last stop # of epochs
    if exist('costValMin', 'var') && (countEpoch > stop) ...
            && all( cost(3, (countEpoch+1-stop):countEpoch ) > costValMin );
        fprintf('\n\nEarly stopping - validation cost has not improved in %d epochs.\n', stop);
        return      % EXIT NEURAL_NET
    end
        
    % INCREMENT epoch counter
    countEpoch = countEpoch +1;
 
end                                             % END WHILE
 
if accTrn == 1
    fprintf('\n\nEarly stopping - 100%% accuracy acheived!\n');
end
 
end     % END function
