function practical2b

%The goal of this part of the practical is to take a real image containing
%a planar black square and figure out the transformation between the square
%and the camera.  We will then draw a wire-frame cube with it's base
%corners at the corner of the square.  You should use this
%template for your code and fill in the missing sections marked "TO DO"

%load in image 
im = imread('test104.jpg');

%define points on image
xImCart = [  140.3464  212.1129  346.3065  298.1344   247.9962;...
             308.9825  236.7646  255.4416  340.7335   281.5895];
         
%define 3D points of plane
XCart = [-50 -50  50  50 0 ;...
          50 -50 -50  50 0;...
           0   0   0   0 0];

%We assume that the intrinsic camera matrix K is known and has values
K = [640  0    320;...
     0    640  240;
     0    0    1];

%draw image and 2d points
figure; set(gcf,'Color',[1 1 1]);
imshow(im); axis off; axis image; hold on;
plot(xImCart(1,:),xImCart(2,:),'r.','MarkerSize',10);
       
%TO DO Use your routine to calculate TEst, the extrinsic matrix relating the
%plane position to the camera position.
TEst = estimatePlanePose(xImCart, XCart, K);


%define 3D points of plane
XWireFrameCart = [-50 -50  50  50 -50 -50  50  50;...
                   50 -50 -50  50  50 -50 -50  50;...
                    0   0   0   0 -100 -100 -100 -100];

%TO DO Draw a wire frame cube, by projecting the vertices of a 3D cube
%through the projective camera and drawing lines betweeen the resulting 2d image
%points
xImCartEst = projectiveCamera(K,TEst,XWireFrameCart);

%draw image and 2d points
figure; set(gcf,'Color',[1 1 1]);
imshow(im); axis off; axis image; hold on;
plot(xImCartEst(1,:),xImCartEst(2,:),'r.','MarkerSize',10);

%draw lines
for cPoint = 1:4
    %plot a green line between each pair of points
    pt2 = mod(cPoint,4)+1 ;
    plot([xImCartEst(1,cPoint) xImCartEst(1,pt2)],[xImCartEst(2,cPoint) xImCartEst(2,pt2)],'b-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint+4) xImCartEst(1,pt2+4)],[xImCartEst(2,cPoint+4) xImCartEst(2,pt2+4)],'b-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint) xImCartEst(1,cPoint+4)],[xImCartEst(2,cPoint) xImCartEst(2,cPoint+4)],'b-','LineWidth',2);
end;

% plot([xImCartEst(1,1) xImCartEst(1,2)],[xImCartEst(2,1) xImCartEst(2,2)],'b-','LineWidth',2);
% plot([xImCartEst(1,1) xImCartEst(1,5)],[xImCartEst(2,1) xImCartEst(2,5)],'b-','LineWidth',2);
% plot([xImCartEst(1,5) xImCartEst(1,6)],[xImCartEst(2,5) xImCartEst(2,6)],'b-','LineWidth',2);
% plot([xImCartEst(1,2) xImCartEst(1,6)],[xImCartEst(2,2) xImCartEst(2,6)],'b-','LineWidth',2);
% plot([xImCartEst(1,2) xImCartEst(1,3)],[xImCartEst(2,2) xImCartEst(2,3)],'b-','LineWidth',2);
% plot([xImCartEst(1,6) xImCartEst(1,7)],[xImCartEst(2,6) xImCartEst(2,7)],'b-','LineWidth',2);
% plot([xImCartEst(1,3) xImCartEst(1,7)],[xImCartEst(2,3) xImCartEst(2,7)],'b-','LineWidth',2);
% plot([xImCartEst(1,3) xImCartEst(1,4)],[xImCartEst(2,3) xImCartEst(2,4)],'b-','LineWidth',2);
% plot([xImCartEst(1,7) xImCartEst(1,8)],[xImCartEst(2,7) xImCartEst(2,8)],'b-','LineWidth',2);
% plot([xImCartEst(1,4) xImCartEst(1,8)],[xImCartEst(2,4) xImCartEst(2,8)],'b-','LineWidth',2);
% plot([xImCartEst(1,4) xImCartEst(1,1)],[xImCartEst(2,4) xImCartEst(2,1)],'b-','LineWidth',2);
% plot([xImCartEst(1,8) xImCartEst(1,4)],[xImCartEst(2,8) xImCartEst(2,4)],'b-','LineWidth',2);


end

%QUESTIONS TO THINK ABOUT...

%Do the results look realistic?
%If not, then what factors do you think might be causing this?

%goal of function is to project points in XCart through projective camera
%defined by intrinsic matrix K and extrinsic matrix T.
function xImCart = projectiveCamera(K,T,XCart)

%TO DO convert Cartesian 3d points XCart to homogeneous coordinates XHom
XHom = [XCart; ones(1,size(XCart,2))];

%TO DO apply extrinsic matrix to XHom to move to frame of reference of
%camera
XHom = T*XHom;

%TO DO project points into normalized camera coordinates xCamHom by (achieved by
%removing fourth row)
XCamHom = XHom(1:3,:);

%TO DO move points to image coordinates xImHom by applying intrinsic matrix
xImHom = K*XCamHom;

%TO DO convert points back to Cartesian coordinates xImCart
xImCart = xImHom(1:2,:)./repmat(xImHom(3,:),2,1);
end 
%==========================================================================
%==========================================================================

%goal of function is to estimate pose of plane relative to camera
%(extrinsic matrix) given points in image xImCart, points in world XCart
%and intrinsic matrix K.
function T = estimatePlanePose(xImCart,XCart,K)


%TO DO Convert Cartesian image points xImCart to homogeneous representation
%xImHom
xImHom = [xImCart ; ones(1,size(xImCart,2))];

%TO DO Convert image co-ordinates xImHom to normalized camera coordinates
%xCamHom
xCamHom = inv(K)*xImHom;

%TO DO Estimate homography H mapping homogeneous (x,y)
%coordinates of positions in real world to xCamHom.  Use the routine you wrote for
%Practical 1B.
% CalcBestHomography apply to cartesian coordinate :
xCamCart = xCamHom(1:2,:)./repmat(xCamHom(3,:),2,1);
HEst = calcBestHomography(XCart, xCamCart);

%TO DO Estimate first two columns of rotation matrix R from the first two
%columns of H using the SVD

[U,L,V] = svd(HEst(:,1:2));
R12 = U*[1 0; 0 1; 0 0]*V';

%TO DO Estimate the third column of the rotation matrix by taking the cross
%product of the first two columns
R3 = cross(R12(:,1), R12(:,2));

%TO DO Check that the determinant of the rotation matrix is positive - if
%not then multiply last column by -1.
R = [R12 R3];
if det(R) < 0
    R = [R12 -R3];
end

%TO DO Estimate the translation t by finding the appropriate scaling factor k
%and applying it to the third column of H
% see point (15.47)
k = (1/6)* sum(sum( HEst(:,1:2)./R(:,1:2)));
HEst(:,3) = HEst(:,3)/k;

%TO DO Check whether t_z is negative - if it is then multiply t by -1 and
%the first two columns of R by -1.
if HEst(3:3) < 0
    HEst(:,3) = -HEst(:,3);
    R(:, 1:2) = -R(:, 1:2);
end

%assemble transformation into matrix form
T = [R HEst(:,3); 0 0 0 1];
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
