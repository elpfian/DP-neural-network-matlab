function res = screen_print(countEpoch, costTrn, corTrn, numTrn, accTrn,...
    costTst, corTst, numTst, accTst, costVal, corVal, numVal, accVal)
% Prints to screen for training, test and validation sets
% countEpoch = current epoch
% cost... = cost for train, test or validation
% cor... = # correct for train, test or validation
% num... = total cases in train, test or validation
% acc... = accuracy for train, test or validation
 
%% Print levels depending on size of training sample
prntLevel = floor(log10(numTrn));
 
switch prntLevel
    case 4 % size >=10,000
        titleStyle = '%4s|%12s%-19s||%12s%-19s||%12s%-19s|\n';
        headStyle = '| %-7s| %-12s| %-7s|';
        if costTrn > 100
            setStyle = '| %6.0f | %5d/%-5d | %6.3f |';
        else
            setStyle = '| %6.3f | %5d/%-5d | %6.3f |';
        end
        hLine = ones(1, 103)*'-';
        
    case 3 % size >= 1,000   
        titleStyle = '%4s|%12s%-17s||%12s%-17s||%12s%-17s|\n';
        headStyle = '| %-7s| %-10s| %-7s|';
        setStyle = '| %6.3f | %4d/%-4d | %6.3f |';
        hLine = ones(1, 97)*'-';
        
    case 2 % size >= 100        
        titleStyle = '%4s|%10s%-17s||%10s%-17s||%10s%-17s|\n';
        headStyle = '| %-7s| %-8s| %-7s|';
        setStyle = '| %6.3f | %3d/%-3d | %6.3f |';
        hLine = ones(1, 91)*'-';
        
    case 1 % size >= 10
        titleStyle = '%4s|%8s%-17s||%8s%-17s||%8s%-17s|\n';
        headStyle = '| %-7s| %-6s| %-7s|';
        setStyle = '| %6.3f | %2d/%-2d | %6.3f |';
        hLine = ones(1, 85)*'-';
        
    case 0 % size >= 0
        titleStyle = '%4s|%8s%-15s||%8s%-15s||%8s%-15s|\n';
        headStyle = '| %-7s| %-4s| %-7s|';
        setStyle = '| %6.3f | %1d/%-1d | %6.3f |';
        hLine = ones(1, 79)*'-';
end

 
%% Title and header rows
if countEpoch == 1
    sp = ' ';
    trChr = 'TRAIN';
    tsChr = 'TEST';
    vlChr = 'VALIDATION';
    fprintf(titleStyle, sp, sp, trChr, sp, tsChr, sp, vlChr);
 
    cstChr = 'Cost';
    corChr = 'Cor';
    accChr = 'Acc';
 
    fprintf(' Ep ');
    fprintf(headStyle, cstChr, corChr, accChr); % Train
    fprintf(headStyle, cstChr, corChr, accChr); % Test
    fprintf(headStyle, cstChr, corChr, accChr); % Validation
    fprintf('\n');
 
    fprintf('%s\n', hLine); % Horizontal line
end
 
%% Single Epoch
epStyle = '%3d ';
 
fprintf(epStyle, countEpoch);
fprintf(setStyle, costTrn, corTrn, numTrn, accTrn); % Train
fprintf(setStyle, costTst, corTst, numTst, accTst); % Test
fprintf(setStyle, costVal, corVal, numVal, accVal); % Validation
fprintf('\n');
end
