function HW2_TrackingAndHomographies


LLs = HW2_Practical9c( 'll' );
LRs = HW2_Practical9c( 'lr' );
ULs = HW2_Practical9c( 'ul' );
URs = HW2_Practical9c( 'ur' );

close all;

% Load frames from the whole video into Imgs{}.
% This is really wasteful of memory, but makes subsequent rendering faster.
LoadVideoFrames

% Coordinates of the known target object (a dark square on a plane) in 3D:
XCart = [-50 -50  50  50;...
          50 -50 -50  50;...
           0   0   0   0];

% These are some approximate intrinsics for this footage.
K = [640  0    320;...
     0    512  256;
     0    0    1];

% Define 3D points of wireframe object.
XWireFrameCart = [-50 -50  50  50 -50 -50  50  50;...
                   50 -50 -50  50  50 -50 -50  50;...
                    0   0   0   0 -100 -100 -100 -100];
 
hImg = figure;
       
% ================================================
for iFrame = 1:numFrames
    xImCart = [LLs(iFrame,:)' ULs(iFrame,:)' URs(iFrame,:)' LRs(iFrame,:)'];
    xImCart = circshift( xImCart, 1);

    % To get a frame from footage 
    im = Imgs{iFrame};

    % Draw image and 2d points
    set(0,'CurrentFigure',hImg);
    set(gcf,'Color',[1 1 1]);
    imshow(im); axis off; axis image; hold on;
    plot(xImCart(1,:),xImCart(2,:),'r.','MarkerSize',15);


    %TO DO Use your routine to calculate TEst the extrinsic matrix relating the
    %plane position to the camera position.
    TEst = estimatePlanePose(xImCart, XCart, K);

    %TO DO Draw a wire frame cube, by projecting the vertices of a 3D cube
    %through the projective camera, and drawing lines betweeen the 
    %resulting 2d image points
    hold on;
    
    % TO DO: Draw a wire frame cube using data XWireFrameCart. You need to
    % 1) project the vertices of a 3D cube through the projective camera;
    xImCartEst = projectiveCamera(K,TEst,XWireFrameCart);
    % 2) draw lines betweeen the resulting 2d image points.
    %draw image and 2d points
imshow(im); axis off; axis image; hold on;
plot(xImCartEst(1,:),xImCartEst(2,:),'r.','MarkerSize',10);

%draw lines for the cube
for cPoint = 1:4
    %plot a green line between each pair of points
    pt2 = mod(cPoint,4)+1 ;
    plot([xImCartEst(1,cPoint) xImCartEst(1,pt2)],[xImCartEst(2,cPoint) xImCartEst(2,pt2)],'b-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint+4) xImCartEst(1,pt2+4)],[xImCartEst(2,cPoint+4) xImCartEst(2,pt2+4)],'b-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint) xImCartEst(1,cPoint+4)],[xImCartEst(2,cPoint) xImCartEst(2,cPoint+4)],'b-','LineWidth',2);
end;
    % Note: CONDUCT YOUR CODE FOR DRAWING XWireFrameCart HERE
    
    
    hold off;
    drawnow;
    
%     Optional code to save out figure
%pngFileName = sprintf( '%s_%.5d.png', 'myOutput', iFrame );
%print( gcf, '-dpng', '-r80', pngFileName ); % Gives 640x480 (small) figure

    
end % End of loop over all frames.
end
% ================================================

% TO DO: QUESTIONS TO THINK ABOUT...

% Q: Do the results look realistic?
% If not then what factors do you think might be causing this


% TO DO: your routines for computing a homography and extracting a 
% valid rotation and translation GO HERE. Tips:
%
% - you may define functions for T and H matrices respectively.
% - you may need to turn the points into homogeneous form before any other
% computation. 
% - you may need to solve a linear system in Ah = 0 form. Write your own
% routines or using the MATLAB builtin function 'svd'. 
% - you may apply the direct linear transform (DLT) algorithm to recover the
% best homography H.
% - you may explain what & why you did in the report.

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

