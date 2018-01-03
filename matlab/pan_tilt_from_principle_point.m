function pt = pan_tilt_from_principle_point(pp, pp_ptz, p)
% generate pan_tilt sample points using ptz ground truth
% pp: pincipal point
% pp_ptz: pincipal point pan, tilt zoom, eg. ptz of the image
% p: a point in the image, unit pixel
% pt: return value, pan and tilt

dx = p(1) - pp(1);
dy = p(2) - pp(2);
pan_pp = pp_ptz(1);
tilt_pp = pp_ptz(2);
fl = pp_ptz(3);
delta_pan = atan2(dx, fl) * 180/pi;
delta_tilt = atan2(dy, fl) * 180/pi;
pt = zeros(2, 1);
pt(1) = pan_pp + delta_pan;
pt(2) = tilt_pp + delta_tilt;
end



