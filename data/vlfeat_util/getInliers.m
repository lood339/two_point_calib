% get inlier index of matchings
function [matches_geo, H] = getInliers(frames1, descrs1, frames2, descrs2)
% descrs1 and descrs2: sift feature from vlfeat
% matches_geo: return value matched index in 
[nn, dist2] = findNeighbours(descrs1, descrs2, 2) ;
matches_2nn = [1:size(nn,2), 1:size(nn,2) ; nn(1,:), nn(2,:)] ;

%% -------------------------------------------------------------------
%               Stage I.D: Better matching w/ geometric transformation
% --------------------------------------------------------------------

[inliers, H] = geometricVerification(frames1, frames2, matches_2nn, 'numRefinementIterations', 8) ;
matches_geo = matches_2nn(:, inliers) ;
end