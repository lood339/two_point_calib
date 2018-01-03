% demonstation of calibration from two points using synthetic data

clear
close all;

% ground truth, ptz
ptz = [0.520949	 -11.433174	 1919.092246]';
pp = [1280.0/2, 720.0/2]';

w = 1280;
h = 720;

% random sample two points
N = 100;
errors = zeros(N, 3);
for i = [1:N]
    % generate two image location
    point1 = zeros(2, 1);
    point2 = zeros(2, 1);
    point1(1) = randi(w, 1) + rand();
    point1(2) = randi(h, 1) + rand();
    point2(1) = randi(w, 1) + rand();
    point2(2) = randi(h, 1) + rand();
    
    % pan tilt values of sampled image location
    % In real problems, pan, tilt values are from the random forest
    pt1 = pan_tilt_from_principle_point(pp, ptz, point1);   
    pt2 = pan_tilt_from_principle_point(pp, ptz, point2);       

    % main algorithm
    estimated_ptz = ptz_from_two_point(pp, pt1, pt2, point1, point2);
    errors(i, :) = abs(estimated_ptz - ptz);    
end

mean(errors)
std(errors)




