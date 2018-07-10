function [wCur, bCur] = neural_net( inputs, targets, nodeLayers, numEpochs, batchSize, eta )
% Learns a neural network. Writes performance to file 'results.txt'. 
% No output returned.
%
% FUNCTION VARIABLES
% inputs = matrix, col for each example and row for each input feature
% targets = matrix, col for each example and row for each output feature
% nodeLayers = vector, # of nodes in each layer
% numEpochs = scalar, # of epochs to run
% batchSize = scalar, # of examples in mini-batch
% eta = scalar, learning rate
%
% REQUIRED FUNCTION FILES: init_weights, init_biases, back_prop, sigma,
% meanSqErr
 
% INTERMEDIATE VARIABLES
trainSize = size( inputs, 2 );
netDepth = size( nodeLayers, 2);
numBatches = ceil( trainSize/batchSize );
 
% INITIALIZE weights and biases
wCur = init_weights( nodeLayers );
bCur = init_biases( nodeLayers );
 
% INITIALIZE loop conditionals
countEpoch = 1;     % initialize epoch counter
acc = 0;            % initialize accuracy
 
% OPEN OUTPUT FILE
fName = 'results.txt';
fileID = fopen(fName, 'w');
 
% MAIN LOOP
% Stop when numEpochs completed or accuracy = 100%
while (countEpoch <= numEpochs) && (acc < 1);       % START WHILE loop
   
    % Index to randomize training set order for each epoch
    shuffInd = randperm( trainSize );
 
    batchStart = 1;     % initialize batchStart pointer
    
    % LOOP OVER EPOCH
    % Stop after all records used
    for batch = 1:numBatches        % START epoch
        
        % GENERATE MINI-BATCH
        batchEnd = batchStart + batchSize -1;   % create batchEnd pointer
 
        % select vector of indices batchInd from shuffled index
        if batchEnd > size( shuffInd );     % last batch
            batchInd = shuffInd( batchStart: end );
        else                                % all other batches
            batchInd = shuffInd( batchStart:( batchEnd ) );
        end
        
        % select inputs, targets for batch using batchInd
        x_batch = inputs( :, batchInd );    % all rows, shuffled cols
        y_batch = targets( :, batchInd );   % all rows, shuffled cols
 
        batchStart = batchStart + batchSize;% increment batchStart pointer
        
        % BACKPROPAGATION
        % Run for each mini-batch, returns [new_weights, new_biases]
        % Use "size(batchInd, 2)" instead of batch size in case
        %   last batch is different size
        
        temp = {};      % clear temp result
        
        [temp{1:2}] = back_prop(x_batch, y_batch,...
            wCur, bCur, netDepth, eta, size(batchInd, 2) );
        
        wCur = temp{1};
        bCur = temp{2};
                
    end                             % END epoch
    
    
    % GENERATE NETWORK OUTPUT 
    % For entire training set with updated weights/biases
    % Initialize activation and weighted inputs
    activation = sigma(inputs);
    zInput = [];
 
    % Feedforward over entire network. Final activation = network output.
    % Can overwrite activation, zInput as only need final activation
    for i = 2:netDepth
        
        % calculate weighted input for layer i
        zInput = wCur{i}*activation + repmat( bCur{i}, 1, trainSize );
        
        % calculate activation for layer i
        activation = sigma( zInput );
 
    end
        
    % CALCULATE AND PRINT
    % MSE, Correct Classifications, Accuracy
    mse = meanSqErr( activation, targets );
    correct = sum( all( targets == round(activation), 1 ) );
    acc = correct / trainSize;
    
    % PRINT to fileID (named above while loop)
    style = 'Epoch %3d \t MSE: %5.4f \t Correct: %d / %d \t Acc: %.3f\r\n';
    fprintf(fileID, style, countEpoch, mse, correct, trainSize, acc);
    
    
    % INCREMENT epoch counter
    countEpoch = countEpoch +1;
 
end                                             % END WHILE
 
% Close results file
fclose(fileID);
 
end     % END function
