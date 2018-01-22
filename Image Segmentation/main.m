%% load training set
[RGBApple, RGBNonApple, priorApple, priorNonApple] = createTrainingSet;

%% Train a model : mixture of 3 Gaussians.
nIter = 15; %number of Iteration for the E-M Algorithm.
nGauss = 3;

disp('Fitting Apple pixels for a mixture of 3 gaussians.')
[weightApple1, meanApple1, covApple1] = fitMixGauss(RGBApple,nGauss,nIter);

disp('Fitting Non Apple pixels for a mixture of 3 gaussians.')
[weightNonApple1, meanNonApple1, covNonApple1] = fitMixGauss(RGBNonApple,nGauss,nIter);

%% Load validation set
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

%% Model Validation

% Compute posteriors of the validation set images
posteriorApple1 = getPosterior(imV1,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);
posteriorApple2 = getPosterior(imV2,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);
posteriorApple3 = getPosterior(imV3,weightApple1, meanApple1, covApple1,weightNonApple1, meanNonApple1, covNonApple1, priorApple, priorNonApple);

%plot ROC Curve and give best threshold
[X3, Y3] = ROCCurve2(GT1, GT2, GT3, posterior1, posterior2, posterior3);
k3 = bestThreshold(X3,Y3);
figure; plot(X3,Y3, 'r-'); legend('mixture of 3 Gaussians'); xlabel('FP Rate'); ylabel('TP Rate');

