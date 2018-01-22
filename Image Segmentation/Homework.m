function r=Homework
%% (creating training set)
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';

% Loading the photos :
% curI is now a double-precision 3D matrix of size (width x height x 3). 
% Each of the 3 color channels is now in the range [0.0, 1.0].
curI1 = double(imread(  Iapples{1}   )) / 255;
curI2 = double(imread(  Iapples{2}   )) / 255;
curI3 = double(imread(  Iapples{3}   )) / 255;

% Loading the binary masks 
curImask1 = imread(  IapplesMasks{1}   );
curImask2 = imread(  IapplesMasks{2}   );
curImask3 = imread(  IapplesMasks{3}   );

% Picked red channel arbitrarily. could have picked green or blue.
% The masks contains 1 if the corresponding pixel is a  part of an apple, 0 else
curImask1 = curImask1(:,:,1) > 128;
curImask2 = curImask2(:,:,1) > 128;
curImask3 = curImask3(:,:,1) > 128;

%Thanks to the mask, we can now build 2 set : one set of "apple pixel" and
%another one of "non-apple pixels" : each data of the set is a 3x1 vector 
%containing the RGB details of the pixel

%Initialization of the training set.

% counting the number of data for each set.
nApple = 0; % number of "apple pixel"
nNApple = 0; %nb of "non apple pixel"

[N1, M1] = size(curImask1);
[N2, M2] = size(curImask2);
[N3, M3] = size(curImask3);

%counting nb of apple pixel in curI1
nApple = nApple + sum(sum(curImask1));
nNApple = nNApple + N1*M1 - sum(sum(curImask1));

%counting nb of apple pixel in curI2
nApple = nApple + sum(sum(curImask2));
nNApple = nNApple + N2*M2 - sum(sum(curImask2));

%counting nb of apple pixel in curI3
nApple = nApple + sum(sum(curImask3));
nNApple = nNApple + N3*M3 - sum(sum(curImask3));

RGBApple = zeros(3,nApple);
RGBNonApple = zeros(3,nNApple);

%Creating training set
k=1;
l=1;
% Adding data from curI1
for i=1:N1
    for j=1:M1
        if curImask1(i,j) == 1
            RGBApple(:,k) = curI1(i,j,:);
            k = k +1;
        else
            RGBNonApple(:,l) = curI1(i,j,:);
            l = l +1;
        end
    end
end

% Adding data from curI2
for i=1:N2
    for j=1:M2
        if curImask2(i,j) == 1
            RGBApple(:,k) = curI2(i,j,:);
            k = k +1;
        else
            RGBNonApple(:,l) = curI2(i,j,:);
            l = l +1;
        end
    end
end
% 
%Adding data from curI3
for i=1:N3
    for j=1:M3
        if curImask3(i,j) == 1
            RGBApple(:,k) = curI3(i,j,:);
            k = k +1;
        else
            RGBNonApple(:,l) = curI3(i,j,:);
            l = l +1;
        end
    end
end

% priors defined according to the training set
priorApple = nApple/(nApple + nNApple);
priorNonApple = nNApple/(nApple + nNApple);

%% Fitting mixture of gaussians to RGB data (train and validation)

% the following section is used for optimizing the model : find the best
% number of estimated function and the suited threshold.
% number of estimate gaussians
%nGaussEsts = [2:2:10];
nIter = 15 ;

%Creating validation set
imageValidation = cell(3,1);
imageVGroundTruch = cell(3,1);

imageValidation{1} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg';
imageValidation{2} = 'testApples/AppleValidation1.jpg';
imageValidation{3} = 'testApples/AppleValidation2.jpg';

imageVGroundTruch{1} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.png';
imageVGroundTruch{2} = 'testApples/AppleValidation1GT.jpg';
imageVGroundTruch{3} = 'testApples/AppleValidation1GT.jpg';

imV1 = double(imread(imageValidation{1})) / 255;
imV2 = double(imread(imageValidation{2})) / 255;
imV3 = double(imread(imageValidation{3})) / 255;

% associated ground truth
GT1 = rgb2gray(imread(imageVGroundTruch{1}));
GT2 = rgb2gray(imread(imageVGroundTruch{2}));
GT3 = rgb2gray(imread(imageVGroundTruch{3}));

%==========================================================================
%==========================================================================

% The following commented section should be used to compare the prediction
% of several mixtures of different number of gaussians, and giving the best
% threshold for each. By lack of time, I have only compare 2 mixtures (see
% below)

%figure;
%colorArray = ['y-', 'm-', 'c-', 'r-', 'g-']
%thresholdArray = zeros(1,size(nGaussEsts,2));

% for i=1:size(nGaussEsts,2)
%     
%     sprintf('Fitting Apple pixels for %d estimated gaussians.', i);
%     [weightApple, meanApple, covApple] = fitMixGauss(RGBApple,nGaussEsts(i),nIter);
% 
%     sprintf('Fitting Non Apple pixels for %d estimated gaussians.', i);
%     [weightNonApple, meanNonApple, covNonApple] = fitMixGauss(RGBNonApple,nGaussEsts(i),nIter);
%     
%     % Model Validation : pick the model with the best accuracy
%     % Compute posterior for each image and saving it
%     posteriorApple1 = getPosterior(imV1,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple, priorApple, priorNonApple);
%     posteriorApple2 = getPosterior(imV2,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple, priorApple, priorNonApple);
%     posteriorApple3 = getPosterior(imV3,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple, priorApple, priorNonApple);
%     
%     filename1 = sprintf('testApples/postAp1_%d.jpg',i);
%     filename2 = sprintf('testApples/postAp2_%d.jpg',i);
%     filename3 = sprintf('testApples/postAp3_%d.jpg',i);
%     
%     imwrite(posteriorApple1,filename1);
%     imwrite(posteriorApple2,filename2);
%     imwrite(posteriorApple3,filename3);
%     
%     % We now have the posterior for different mixture and for each Gaussian
%     %Plot ROC Curve for each mixture 
%     posterior1 = imread(filename1);
%     posterior2 = imread(filename2);
%     posterior3 = imread(filename3);
%     
%     % plot ROC Curve and determines best threshold for each gaussian
%     [X, Y] = ROCCurve2(GT1, GT2, GT3, posterior1, posterior2, posterior3);
%     plot(X,Y,colorArray(i)); legend(sprintf('%d gaussians',i)); hold on;
%     thresholdArray(i) = bestThreshold(X,Y)
% end

%==========================================================================
%==========================================================================

%Compare the prediction error between a mixture of 3 gaussians, and another
%one of 6 gaussians
% Train model
disp('Fitting Apple pixels for a mixture of 3 gaussians.')
[weightApple1, meanApple1, covApple1] = fitMixGauss(RGBApple,3,nIter);

disp('Fitting Non Apple pixels for a mixture of 3 gaussians.')
[weightNonApple1, meanNonApple1, covNonApple1] = fitMixGauss(RGBNonApple,3,nIter);

disp('Fitting Apple pixels for a mixture of 6 gaussians.')
[weightApple2, meanApple2, covApple2] = fitMixGauss(RGBApple,6,nIter);

disp('Fitting Non Apple pixels for a mixture of 6 gaussians.')
[weightNonApple2, meanNonApple2, covNonApple2] = fitMixGauss(RGBNonApple,6,nIter);

% Model Validation

%Computing posteriors of the images
%Mixture of 3 gaussians
posteriorApple13 = getPosterior(imV1,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);
posteriorApple23 = getPosterior(imV2,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);
posteriorApple33 = getPosterior(imV3,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);

%Mixture of 6 gaussians
posteriorApple16 = getPosterior(imV1,weightApple2, meanApple2, covApple2,weightNonApple2, meanNonApple2, covNonApple2, priorApple, priorNonApple);
posteriorApple26 = getPosterior(imV2,weightApple2, meanApple2, covApple2,weightNonApple2, meanNonApple2, covNonApple2, priorApple, priorNonApple);
posteriorApple36 = getPosterior(imV3,weightApple2, meanApple2, covApple2,weightNonApple2, meanNonApple2, covNonApple2, priorApple, priorNonApple);

imwrite(posteriorApple13,'testApples/PosteriorAppleTest13.jpg');
imwrite(posteriorApple23,'testApples/PosteriorAppleTest23.jpg');
imwrite(posteriorApple33,'testApples/PosteriorAppleTest33.jpg');
imwrite(posteriorApple16,'testApples/PosteriorAppleTest16.jpg');
imwrite(posteriorApple26,'testApples/PosteriorAppleTest26.jpg');
imwrite(posteriorApple36,'testApples/PosteriorAppleTest36.jpg');

posterior13 = imread('testApples/PosteriorAppleTest13.jpg');
posterior23 = imread('testApples/PosteriorAppleTest23.jpg');
posterior33 = imread('testApples/PosteriorAppleTest33.jpg');
posterior16 = imread('testApples/PosteriorAppleTest16.jpg');
posterior26 = imread('testApples/PosteriorAppleTest26.jpg');
posterior36 = imread('testApples/PosteriorAppleTest36.jpg');

%plot ROC Curve and give best threshold
[X3, Y3] = ROCCurve2(GT1, GT2, GT3, posterior13, posterior23, posterior33);
[X6, Y6] = ROCCurve2(GT1, GT2, GT3, posterior16, posterior26, posterior36);
k3 = bestThreshold(X3,Y3)
k6 = bestThreshold(X6,Y6)
figure; plot(X3,Y3, 'r-'); legend('mixture of 3 Gaussians'); xlabel('FP Rate'); ylabel('TP Rate'); hold on;
plot(X6,Y6,'g-');legend('mixture of 6 Gaussians');

%==========================================================================
%==========================================================================

%% Calculate posteriors (testing model : run model on unseen data)

% % Uncomment this section to run classification

% % Fitting 'best model' of mixture of gaussians (chosen by comparing the
% % ROC curve during the previous section
% nGaussEst = 3; %replace value by the 'best' number of estimated gaussians
% nIter = 20;
% %fit one Gaussian model for Apple pixels
% disp('Fitting Apple pixels.')
% [weightApple, meanApple, covApple] = fitMixGauss(RGBApple,nGaussEst,nIter);
% 
% %fit another Gaussian model for non-Apple pixels
% disp('Fitting Non Apple pixels.')
% [weightNonApple, meanNonApple, covNonApple] = fitMixGauss(RGBNonApple,nGaussEst,nIter);

% % replace value by the 'best' threshold given the number of estimated gaussian
% threshold = 114; 

% % run through the pixels in the image and classify them as being Apple or
% % non Apple

% imageTest = cell(2,1);
% 
% imageTest{1} = 'testApples/Apples_by_MSR_MikeRyan_flickr.jpg';
% imageTest{2} = 'testApples/audioworm-QKUJj2wmxuI-original.jpg';
% 
% im1 = double(imread(imageTest{1})) / 255;
% im2 = double(imread(imageTest{2})) / 255;
% 
% figure; set(gcf,'Color',[1 1 1]);
% subplot(2,3,1); imagesc(im1); title('Original'); axis off; axis image;
% subplot(2,3,4); imagesc(im2); title('Original'); axis off; axis image;
% drawnow;
% 
% posteriorApple1 = getPosterior(im3,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple, priorApple, priorNonApple);
% 
% %draw skin posterior
% clims = [0, 1];
% subplot(2,3,2); imagesc(posteriorApple1, clims); title('posterior'); colormap(gray); axis off; axis image;
% 
% posteriorApple2 = getPosterior(im2,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple,priorApple, priorNonApple);
% 
% %draw skin posterior
% clims = [0, 1];
% subplot(2,3,5); imagesc(posteriorApple2, clims); title('posterior'); colormap(gray); axis off; axis image;

% %Classify the pixels :
% [n m] = size(im1);
% clf1 = zeros(n,m);
% for i=1:m
%     for j=1:n
%         if posteriorApple1(i,j) >= threshold
%             clf(i,j) = 255;
%         end
%     end
% end

% [n m] = size(im2);
% clf2 = zeros(n,m);
% for i=1:m
%     for j=1:n
%         if posteriorApple2(i,j) >= threshold
%             clf(i,j) = 255;
%         end
%     end
% end
% subplot(2,3,3); imagesc(clf1); title('classification');
% subplot(2,3,6); imagesc(clf2); title('classification');
end

%% fit the model to a dataset
function [weight, mean, cov] = fitMixGauss(data,k,nIter)
        
[nDim, nData] = size(data);

%MAIN E-M ROUTINE 
%there are nData data points, and there is a hidden variable associated
%with each.  If the hidden variable is 0 this indicates that the data was
%generated by the first Gaussian.  If the hidden variable is 1 then this
%indicates that the hidden variable was generated by the second Gaussian
%etc.

postHidden = zeros(k, nData);

%in the E-M algorithm, we calculate a complete posterior distribution over
%the (nData) hidden variables in the E-Step.  In the M-Step, we
%update the parameters of the Gaussians (mean, cov, w).  

% we will initialize the values to random values
mixGaussEst.d = nDim;
mixGaussEst.k = k;
mixGaussEst.weight = (1/k)*ones(1,k);
mixGaussEst.mean = 2*randn(nDim,k);
for cGauss =1:k
    mixGaussEst.cov(:,:,cGauss) = (0.5+1.5*rand(1))*eye(nDim,nDim);
end


%calculate current likelihood
logLike = getMixGaussLogLike(data,mixGaussEst);
fprintf('Log Likelihood Iter 0 : %4.3f\n',logLike);



for cIter = 1:nIter
   %Expectation step
   for cData = 1:nData
        %calculate posterior probability that thisData point came from each of the Gaussians
        thisData = data(:,cData);
        for i=1:mixGaussEst.k
            postHidden(i,cData) = mixGaussEst.weight(i)*getGaussProb(thisData, mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i));
        end
        postHidden(:,cData) = postHidden(:,cData)/sum(postHidden(:,cData));
   end
   
   
   %Maximization Step
   
   %for each constituent Gaussian
   for cGauss = 1:mixGaussEst.k
        %Update weighting parameters mixGauss.weight based on the total
        %posterior probability associated with each Gaussian. 
        mixGaussEst.weight(cGauss) = sum(postHidden(cGauss,:))/sum(sum(postHidden)); 
   
        %Update mean parameters mixGauss.mean by weighted average
        %where weights are given by posterior probability associated with
        %Gaussian.
        mixGaussEst.mean(:,cGauss) =  sum(repmat(postHidden(cGauss,:),3,1).*data,2)/sum(postHidden(cGauss,:)) ;
        
        %Update covarance parameter based on weighted average of
        %square distance from update mean, where weights are given by
        %posterior probability associated with Gaussian
        rCov = zeros(nDim);
        for i=1:nData
           rCov = rCov + postHidden(cGauss,i)*(data(:,i)-mixGaussEst.mean(cGauss))*(data(:,i)-mixGaussEst.mean(cGauss))';
        end
        mixGaussEst.cov(:,:,cGauss) = rCov/sum(postHidden(cGauss,:));
   end
  
   %calculate the log likelihood
   logLike = getMixGaussLogLike(data,mixGaussEst);
   fprintf('Log Likelihood After M-Step Iter %d : %4.3f\n',cIter,logLike);

   %calculate the bound
%    bound = getMixGaussBound(data,mixGaussEst,postHidden);
%    fprintf('Bound After M-Step Iter %d : %4.3f\n',cIter,bound); 
   
end

weight = mixGaussEst.weight ;
mean = mixGaussEst.mean ;
cov = mixGaussEst.cov ;

end

%% the goal of this subroutine is to calculate the log likelihood for the whole
%data set under a mixture of Gaussians model. We calculate the log as the
%likelihood will probably be a very small number that Matlab may not be
%able to represent.
function logLike = getMixGaussLogLike(data,mixGaussEst)

%find total number of data items
[~, nData] = size(data);

%initialize log likelihoods
logLike = 0;

%run through each data item
for cData = 1:nData
    thisData = data(:,cData);    
    %calculate likelihood of this data point under mixture of Gaussians model
    like = 0;
    
    for i=1:mixGaussEst.k    
        like = like + mixGaussEst.weight(i)*getGaussProb(thisData, mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i));
    end
    
    %add to total log like
    logLike = logLike+log(like);        
end
end

%% subroutine to return gaussian probabilities
function prob = getGaussProb(x,mean,var)
[nDim ~] = size(x);
A = 1/sqrt(det(var));
B = ((x-mean)')*inv(var)*(x-mean);
prob = (A/sqrt((2*pi)^nDim))*exp(-0.5*B);
end

%% calculate the bound
function bound = getMixGaussBound(data,mixGaussEst,responsibilities)

%find total number of data items
nData = size(data,2);

%initialize bound
bound = 0;

%run through each data item
for cData = 1:nData
    %extract this data
    thisData = data(:,cData);    
    %extract this q(h)
    thisQ = responsibilities(:,cData);
    
    %TO DO - calculate contribution to bound of this datapoint
    %Replace this
    boundValue= 0;
    for i=1:mixGaussEst.k
       boundValue = boundValue + thisQ(i)*log(mixGaussEst.weight(i)*getGaussProb(thisData, mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i))/thisQ(i)); 
    end
    
    %add to total log like
    bound = bound+boundValue;        
end
end


%% Get posterior for an image
function posterior = getPosterior(im,weightApple, meanApple, covApple,weightNonApple, meanNonApple, covNonApple,priorApple, priorNonApple)
    [imY, imX, ~] = size(im);
    posterior = zeros(imY,imX);
    n = size(weightApple,2);
    for cY = 1:imY  
        fprintf('Processing Row %d\n',cY);
        for cX = 1:imX         
        
        %extract this pixel data
        thisPixelData = squeeze(double(im(cY,cX,:)));
        
        likeApple = 0;
        likeNonApple = 0;
        % Likelihood define as the weighted sum of the gaussians
            for i=1:n
                %calculate likelihood of this data given Apple model
                likeApple = likeApple + weightApple(i)*getGaussProb(thisPixelData,meanApple(:,i),covApple(:,:,i));
                %calculate likelihood of this data given non apple model
                likeNonApple = likeNonApple + weightNonApple(i)*getGaussProb(thisPixelData,meanNonApple(:,i),covNonApple(:,:,i));
            end
        
        %calculate posterior with Bayes Rules: 
        posterior(cY,cX) = likeApple*priorApple/(likeApple*priorApple + likeNonApple*priorNonApple);
        end
    end
end


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

%% Get the best threshold
function k = bestThreshold(X,Y)
    %return the indice of the best threshold
    k = 1;
    a = (1-X(1))/Y(1) ;
    dif = abs(1 - a);
    [n, m] = size(X);
    for i=2:m
        b = (1-X(i))/Y(i);
        if abs(1-b)<dif 
            dif = abs(1-b);
            k = i;
        end
    end
end

%% get points [X,Y] to plot ROCCurve for 3 images
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
