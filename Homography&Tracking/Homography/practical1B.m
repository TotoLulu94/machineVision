function practical1B

%the aim of the second part of practical 1 is to use the homography routine
%that you established in the first part of the practical.  We are going to
%make a panorama of several images that are related by a homography.  I
%provide 3 images (one of which is has a large surrounding region) and a
%matching set of points between these images.

%close all open figures
close all;

%load in the required data
load('PracticalDataSm','im1','im2','im3','pts1','pts2','pts3','pts1b');
%im1 is center image with grey background
%im2 is left image 
%pts1 and pts2 are matching points between image1 and image2
%im3 is right image
%pts1b and pts3 are matching points between image 1 and image 3

%show images and points
figure; set(gcf,'Color',[1 1 1]);image(uint8(im1));axis off;hold on;axis image;
plot(pts1(1,:),pts1(2,:),'r.'); 
plot(pts1b(1,:),pts1b(2,:),'m.');
figure; set(gcf,'Color',[1 1 1]);image(uint8(im2));axis off;hold on;axis image;
plot(pts2(1,:),pts2(2,:),'r.'); 
figure; set(gcf,'Color',[1 1 1]);image(uint8(im3));axis off;hold on;axis image;
plot(pts3(1,:),pts3(2,:),'m.'); 

%****TO DO**** 
% calculate homography from pts1 to pts2 and pick the one with the best mean
% squared distance
% We can observe in the practical1 that calcBestHomography can produce
% different homographies (just neet to run the code several times to see
% that). So I am computing different homographies and keeping the "best" one
HEst12Array = cell(20,1);
sqDiffArray = cell(20,1);
for i=1:20
    HEst12Array{i} = calcBestHomography(pts1, pts2); 
    pts2EstHom = HEst12Array{i}*[pts2; ones(1,size(pts2,2))];
    pts2Est = pts2EstHom(1:2,:)./repmat(pts2EstHom(3,:),2,1);
    sqDiffArray{i} = mean(sum((pts2-pts2Est).^2));
end
%Picking the best one
index = 1;
dist = sqDiffArray{1};
for i=2:20
    if sqDiffArray{i} < dist 
        index = i;
        dist = sqDiffArray{i};
    end
end

HEst12 = HEst12Array{index};

%****TO DO**** 
%for every pixel in image 1
    %transform this pixel position with your homography to find where it 
    %is in the coordinates of image 2
    %if it the transformed position is within the boundary of image 2 then 
        %copy pixel colour from image 2 pixel to current position in image 1 
        %draw new image1 (use drawnow to force it to draw)
    %end
%end;
[n, m, ~] = size(im1);
[k, l, ~] = size(im2);
%for every pixel in image 1
for i=1:n
    for j=1:m
        %transform this pixel position with your homography to find where it is in the coordinates of image 2
        %Turn pos into homogeneous representation
        pos = [ j ; i; 1];
        new_posHom = HEst12*pos;
        %convert back to Cartesian
        new_pos = new_posHom(1:2,:)./repmat(new_posHom(3,:),2,1);
        %make sure that position elements are integer
        new_pos = round(new_pos);
        %if it the transformed position is within the boundary of image 2 then
        if and( and(0 < new_pos(2),new_pos(2) <= k), and(0 < new_pos(1), new_pos(1) <= l)) 
            %copy pixel colour from image 2 pixel to current position in image 1
            im1(i,j,:) = im2(new_pos(2), new_pos(1), :);
            disp(i)
            %draw new image1 (use drawnow to force it to draw)
            drawnow;
        end
    end
end

%****TO DO****
%repeat the above process mapping image 3 to image 1.
%Computing different homeography
HEst13Array = cell(20,1);
sqDiffArray = cell(20,1);
for i=1:20
    HEst13Array{i} = calcBestHomography(pts1b, pts3); 
    pts3EstHom = HEst13Array{i}*[pts3; ones(1,size(pts3,2))];
    pts3Est = pts3EstHom(1:2,:)./repmat(pts3EstHom(3,:),2,1);
    sqDiffArray{i} = mean(sum((pts3-pts3Est).^2));
end
%Picking the best one
index = 1;
dist = sqDiffArray{1};
for i=2:20
    if sqDiffArray{i} < dist 
        index = i;
        dist = sqDiffArray{i};
    end
end

HEst13 = HEst13Array{index};

[k, l, ~] = size(im3);
%for every pixel in image 1
for i=1:n
    for j=1:m
        %transform this pixel position with your homography to find where it is in the coordinates of image 2
        %Turn pos into homogeneous representation
        pos = [ j ; i; 1];
        new_posHom = HEst13*pos;
        %convert back to Cartesian
        new_pos = new_posHom(1:2,:)./repmat(new_posHom(3,:),2,1);
        %make sure that position elements are integer
        new_pos = round(new_pos);
        %if it the transformed position is within the boundary of image 3 then
        if and( and(0 < new_pos(2),new_pos(2) <= k), and(0 < new_pos(1), new_pos(1) <= l)) 
            %copy pixel colour from image 3 pixel to current position in image 1
            im1(i,j,:) = im3(new_pos(2), new_pos(1), :);
            disp(i)
            %draw new image1 (use drawnow to force it to draw)
            drawnow
        end
    end
end


figure; set(gcf,'Color',[1 1 1]);image(uint8(im1));axis off;hold on;axis image;

end


%==========================================================================
function H = calcBestHomography(pts1Cart, pts2Cart)

%should apply direct linear transform (DLT) algorithm to calculate best
%homography that maps the points in pts1Cart to their corresonding matchin in 
%pts2Cart

%****TO DO ****: replace this
% H = eye(3);

%**** TO DO ****;
%first turn points to homogeneous
pts1Hom = [pts1Cart; ones(1,size(pts1Cart,2))];
pts2Hom = [pts2Cart; ones(1,size(pts2Cart,2))];

%then construct A matrix which should be (10 x 9) in size
u = pts1Hom(1,:);
v = pts1Hom(2,:);
x = pts2Hom(1,:);
y = pts2Hom(2,:);
n = size(u,2);

if n ~= 1
    % first 2 lines, then doing a loop
    A = [ 0 0 0 -u(1) -v(1) -1 y(1)*u(1) y(1)*v(1) y(1); ...
          u(1) v(1) 1 0 0 0 -x(1)*u(1) -x(1)*v(1) -x(1)];
  

    for i=2:n
        B = [ 0 0 0 -u(i) -v(i) -1 y(i)*u(i) y(i)*v(i) y(i); ...
              u(i) v(i) 1 0 0 0 -x(i)*u(i) -x(i)*v(i) -x(i)];
        A = [ A ; B];
    end 
else
    A = [ 0 0 0 -u(1) -v(1) -1 y(1)*u(1) y(1)*v(1) y(1); ...
          u(1) v(1) 1 0 0 0 -x(1)*u(1) -x(1)*v(1) -x(1)];
end

  
%solve Ah = 0 by calling
h = solveAXEqualsZero(A); %(you have to write this routine too - see below)

%reshape h into the matrix H
%Beware - when you reshape the (9x1) vector x to the (3x3) shape of a homography, you must make
%sure that it is reshaped with the values going first into the rows.  This
%is not the way that the matlab command reshape works - it goes columns
%first.  In order to resolve this, you can reshape and then take the
%transpose
H = transpose(reshape(h, [3,3]));
end

%==========================================================================
function x = solveAXEqualsZero(A);

%****TO DO **** Write this routine 
% To find the solution, we compute the SVD A = ULV.T and choose
% x to be the last column of V
[~,~,V] = svd(A);
x = V(:,end);

end
