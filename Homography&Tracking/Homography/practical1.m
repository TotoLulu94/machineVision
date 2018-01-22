function practical1
%% Initialization 
%The aim of practical 1 is to calculate the homography that best maps two
%sets of points to one another.  We will (eventually) use this for creating
%panoramas, and for calculating the 3d pose of planes.  You should use this
%template for your code and fill in the missing sections marked "TO DO"
    
%close all open figures
close all;

%set of two dimensional Cartesian points
pts1Cart = [  240.5000   16.8351   33.5890  164.2696  149.1911;...
              248.8770  193.5890   251.3901 168.4581  228.7723];

%turn points to homogeneous representation
pts1Hom = [pts1Cart; ones(1,size(pts1Cart,2))]
      
%define a homography
H = [0.6 0.7 -100; 1.0 0.6 50; 0.001 0.002 1.0]

%apply homography to points
pts2Hom = H*pts1Hom

%convert back to Cartesian
pts2Cart = pts2Hom(1:2,:)./repmat(pts2Hom(3,:),2,1)

%add a small amount of noise
noiseLevel = 4.0;
pts2Cart = pts2Cart+noiseLevel*randn(size(pts2Cart));

%draw two set of two dimensional points
%opens figure
figure; set(gcf,'Color',[1 1 1]);
%draw lines between each pair of points
nPoint = size(pts1Cart,2)
for (cPoint = 1:nPoint)
    %plot a green line between each pair of points
    plot([pts1Cart(1,cPoint) pts2Cart(1,cPoint)],[pts1Cart(2,cPoint) pts2Cart(2,cPoint)],'g-');
    %make sure we don't replace with next point
    hold on;
end;

%draws first set of points
plot(pts1Cart(1,:),pts1Cart(2,:),'b.','MarkerSize',20);
%remove axis
set(gca,'Box','Off');

%draws second set of points
%pts1 after homeography
plot(pts2Cart(1,:),pts2Cart(2,:),'r.','MarkerSize',20);

%% Finding Homegraphy to map the 2 sets of points
%now our goal is to transform the first points so that they map to the
%second set of points

%****TO DO****: Fill in the details of this routine 
%At the moment, it just returns and identity matrix (body is below)
HEst = calcBestHomography(pts1Cart, pts2Cart);

%now we will see how well the routine works by applying the mapping and
%measuring the square  distance between the desired and actual positions

%apply homography to points
pts2EstHom = HEst*pts1Hom

%convert back to Cartesian
pts2EstCart = pts2EstHom(1:2,:)./repmat(pts2EstHom(3,:),2,1)

%calculate mean squared distance from actual points
sqDiff = mean(sum((pts2Cart-pts2EstCart).^2))

%draw figure with points before and after
%draw two set of two dimensional points
%opens figure
figure; set(gcf,'Color',[1 1 1]);
%draw lines between each pair of points
nPoint = size(pts1Cart,2)
for (cPoint = 1:nPoint)
    %plot a green line pairs of actual and estimated points
    plot([pts2Cart(1,cPoint) pts2EstCart(1,cPoint)],[pts2Cart(2,cPoint) pts2EstCart(2,cPoint)],'g-');
    %make sure we don't replace with next point
    hold on;
end;

%draws second set of points
plot(pts2Cart(1,:),pts2Cart(2,:),'r.','MarkerSize',20);
%remove axis
set(gca,'Box','Off');

%draws estimated positions of second set of points
plot(pts2EstCart(1,:),pts2EstCart(2,:),'m.','MarkerSize',20);

%% other things **** TO DO ****
% %1. Convince yourself that the homography is ambiguous up to scale (by
% %multiplying it by a constant factor and showing it does the same thing).
% %Can you see why this is mathematically the case?
% 
% %applying the same homography to points scaled by a constant factor 10
% pts2EstHomBis = 10*HEst*pts1Hom
% 
% %convert back to Cartesian
% pts2EstCartBis = pts2EstHomBis(1:2,:)./repmat(pts2EstHomBis(3,:),2,1)
% 
% %calculate mean squared distance from actual points
% sqDiffBis = mean(sum((pts2Cart-pts2EstCartBis).^2))
% 
% %draw figure with points before and after
% %draw two set of two dimensional points
% %opens figure
% figure; set(gcf,'Color',[1 1 1]);
% %draw lines between each pair of points
% nPoint = size(pts1Cart,2)
% for (cPoint = 1:nPoint)
%     %plot a green line pairs of actual and estimated points
%     plot([pts2Cart(1,cPoint) pts2EstCartBis(1,cPoint)],[pts2Cart(2,cPoint) pts2EstCartBis(2,cPoint)],'g-');
%     %make sure we don't replace with next point
%     hold on;
% end;
% 
% %draws second set of points
% plot(pts2Cart(1,:),pts2Cart(2,:),'r.','MarkerSize',20);
% %remove axis
% set(gca,'Box','Off');
% 
% %draws estimated positions of second set of points
% plot(pts2EstCartBis(1,:),pts2EstCartBis(2,:),'m.','MarkerSize',20);


%2. Show empirically that your homography routine can EXACTLY map any four points to any
%other four points

%3. Now move to practical 1b.

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
[U,L,V] = svd(A);
x = V(:,end);

end
