function [RGBApple, RGBNonApple, priorApple, priorNonApple] = createTrainingSet
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

% Creating training set
% index
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

end