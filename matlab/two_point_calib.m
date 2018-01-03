% calibration from two points

clear
close all;

% ground truth, ptz
ptz = [0.520949	 -11.433174	 1919.092246]';
pp = [1280.0/2, 720.0/2]'; % principal point

w = 1280; % image resolution
h = 720;

% random sample two points
N = 100;
errors = zeros(N, 3);
for i = [1:N]
    % generate two random points in the image
    point1 = zeros(2, 1);
    point2 = zeros(2, 1);
    point1(1) = randi(w, 1) + rand();
    point1(2) = randi(h, 1) + rand();
    point2(1) = randi(w, 1) + rand();
    point2(2) = randi(h, 1) + rand();

    % estimate pan, tilt and focal length. The focal length is NOT used in
    % optimization
    ptz1 = pan_tilt_from_principle_point(pp, ptz, point1);   
    ptz2 = pan_tilt_from_principle_point(pp, ptz, point2);
    
    estimated_ptz = ptz_from_two_point(pp, ptz1, ptz2, point1, point2);
    errors(i, :) = abs(estimated_ptz - ptz);    
end

mean(errors)
std(errors)




