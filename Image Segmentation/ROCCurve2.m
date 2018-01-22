%% get points [X,Y] to plot ROCCurve given 3 images
function [X, Y] = ROCCurve2(GT1, GT2, GT3, P1, P2, P3)

[n1, m1] = size(P1);
[n2, m2] = size(P3);
[n3, m3] = size(P2);

% threshold used :
thresholds = [0:1:255];
X =  zeros(1, size(thresholds,2)); % False Positive Rate, absciss of the ROC curve
Y = zeros(1, size(thresholds, 2)); % True Positive Rate, ordinate axis of the ROC curve

% threshold our posterior : if pixels values > k (threshold), then values =
% 255, else 0. Compute number of TP, FP, TN and FN for all the threshold on
% the same image
    for k=1:size(thresholds,2)
        test1 = applyThreshold(P1, thresholds(1,k));
        test2 = applyThreshold(P2, thresholds(1,k));
        test3 = applyThreshold(P3, thresholds(1,k));
        
        TP = 0; % number of True Positive
        FP = 0; % number of False Positive
        TN = 0; % number of True negative
        FN = 0; % number of False negative
        
        % Compare GT1 and P1 once threshold applied.
        for i = 1:n1
            for j=1:m1
                %count number of TP and TN
                if GT1(i,j) == test1(i,j)
                    if GT1(i,j) == 0
                        TN = TN +1;
                    else
                        TP = TP +1;
                    end
                %count number of FP and FN
                else
                    if GT1(i,j) == 0
                        FP = FP +1;
                    else
                        FN = FN +1;
                    end 
                end
            end
        end
        
        % Compare GT2 and P2 once threshold applied.
        for i = 1:n2
            for j=1:m2
                %count number of TP and TN
                if GT2(i,j) == test2(i,j)
                    if GT2(i,j) == 0
                        TN = TN +1;
                    else
                        TP = TP +1;
                    end
                %count number of FP and FN
                else
                    if GT2(i,j) == 0
                        FP = FP +1;
                    else
                        FN = FN +1;
                    end 
                end
            end
        end
        
        % Compare GT1 and P1 once threshold applied.
        for i = 1:n3
            for j=1:m3
                %count number of TP and TN
                if GT3(i,j) == test3(i,j)
                    if GT3(i,j) == 0
                        TN = TN +1;
                    else
                        TP = TP +1;
                    end
                %count number of FP and FN
                else
                    if GT3(i,j) == 0
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