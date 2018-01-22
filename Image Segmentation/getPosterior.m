%% Get posterior of an image : for each pixel, indicate if pixel is an apple pixel or not.
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