function ptz = ptz_from_two_point(principal_point, pan_tilt1, pan_tilt2, point1, point2)
% estimate pan, tilt and zoom from two points
% see Section 3.1 Two-point Algorithm for Data Annotation
% principal_point: [u, v], e.g image center
% pan_tilt1: pan and tile of point1, unit degress
% pan_tilt2: same as pan_tilt1
% point1: point 1 location, unit pixel
% point2: same as pan_tilt2
% return: ptz, estimated pan, tilt and focal length
assert(size(principal_point, 1) == 2);
assert(size(principal_point, 2) == 1);

p1 = point1 - principal_point;
p2 = point2 - principal_point;
a = p1'*p1;
b = p2'*p2;
c = p1'*p2;
z = [0, 0, 1]';
pan1 = pan_tilt1(1);
tilt1 = pan_tilt1(2);
pan2 = pan_tilt2(1);
tilt2 = pan_tilt2(2);
pan_tilt_z = panYTiltX(pan2 - pan1, tilt2 - tilt1) * z;
d = z'*pan_tilt_z;
f = focal_length_from_two_points(a, b, c, d); % this main fail

ptz = zeros(3, 1);
if f == 0
    disp('Warning: estimate focal length failed');
else
    theta1 = pan1 - atan2(p1(1), f)*180.0/pi;
    theta2 = pan2 - atan2(p2(1), f)*180.0/pi;
    ptz(1) = (theta1 + theta2)/2.0;

    phi1 = tilt1 - atan2(p1(2), f)*180.0/pi;
    phi2 = tilt2 - atan2(p2(2), f)*180.0/pi;
    ptz(2) = (phi1 + phi2)/2.0;
    ptz(3) = f;    
end

end

function f = focal_length_from_two_points(a, b, c, d)
% @brief, focal length from two points
% See appendex 6.1 Focal Length from Two Points
t1 = 2*((d^2*a*b - c^2));
t2 = (d^2*(a+b) -2*c)^2 - 4 *(d^2*a*b-c^2)*(d^2-1);
assert(t2 > 0);
t3 = 2*c-d^2*(a+b) + sqrt(t2);

f2 = t1/(t3);
f = 0;
if f2 > 0
    f = sqrt(f2);    
end
assert(f2 > 0);

end