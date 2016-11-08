function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
minErr = 100000.0;

tot_num = length(C_set)*length(sigma_set);
cnt=0;
for c=1:length(C_set)
    c_in = C_set(c);
    for s=1:length(sigma_set)
        cnt=cnt+1;
        s_in = sigma_set(s);
        fprintf("(%d/%d) trainging (C: %f | sigma: %f)", cnt, tot_num, c_in, s_in);
        model = svmTrain(X, y, c_in, @(x1, x2) gaussianKernel(x1, x2, s_in));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        if (err<minErr)
            minErr = err;
            C = c_in;
            sigma = s_in;
        end
        fprintf("Error: %f (MinErr: %f C: %f singma: %f)\n\n\n", err, minErr, C, sigma);
    end
end





% =========================================================================

end
