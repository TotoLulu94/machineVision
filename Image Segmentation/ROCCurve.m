%% get points [X,Y] to plot ROCCurve for a given groundTruth and a given posterior
function [X, Y] = ROCCurve(groundTruth, posterior)
if( size(groundTruth) ~= size(posterior))
    disp('Ground Truth image and Posterior size doesnt match.')
end
[n, m] = size(posterior);

% threshold used :
thresholds = [0:1:255];
X =  zeros(1, size(thresholds,2)); % False Positive Rate, absciss of the ROC curve
Y = zeros(1, size(thresholds, 2)); % True Positive Rate, ordinate axis of the ROC curve

% threshold our posterior : if pixels values > k (threshold), then values =
% 255, else 0. Compute number of TP, FP, TN and FN for all the threshold on
% the same image
    for k=1:size(thresholds,2)
        test = applyThreshold(posterior, thresholds(1,k));
        
        TP = 0; % number of True Positive
        FP = 0; % number of False Positive
        TN = 0; % number of True negative
        FN = 0; % number of False negative
        
        % Compare groundTruth and posterior once threshold applied.
        for i = 1:n
            for j=1:m
                %count number of TP and TN
                if groundTruth(i,j) == test(i,j)
                    if groundTruth(i,j) == 0
                        TN = TN +1;
                    else
                        TP = TP +1;
                    end
                %count number of FP and FN
                else
                    if groundTruth(i,j) == 0
                        FP = FP +1;
                    else
                        FN = FN +1;
                    end 
                end
            end
        end
        X(1,k) = FP/(FP + TN);
        Y(1,k) = TP/(TP + FN);
    end
end

%% Apply threshold to a given posterior : if pixel's value > k, then pixel considered as an Apple pixel, else not.
function im = applyThreshold(posterior, k)
    [n, m] = size(posterior);
    im =  zeros(n,m);
    for i=1:n
        for j=1:m
            if posterior(i,j) >= k %if above threshold, the pixel is considered as an Apple pixel, so white (255). else remains black (0)
                im(i,j) = 255;
            end
        end
    end
    im = uint8(im);
end
