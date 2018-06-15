clear
close all;

% prepare training data for soccer dataset
% this file extract SIFT feature from training sequence and save to .mat
% files

% 1. VL feat directory
vl_feat_dir = '/Users/jimmy/Code/vlfeat/toolbox/vl_setup';
run(vl_feat_dir)

load('annotation/seq1_anno.mat')

addpath('vlfeat_util');

anno = annotation;
base_dir = meta.base_dir; % change this to where the image is
% for example, in my computer
base_dir = '/Users/jimmy/Desktop/two_point_calib_dataset/highlights/seq1';
logo_bbox = meta.logo_bbox;
x = logo_bbox(1);
y = logo_bbox(2);
w = logo_bbox(3);
h = logo_bbox(4);
N = numel(annotation);
for i = [1:N-1]
    image_name1 = fullfile(base_dir, anno(i).image_name);
    camera1 = anno(i).camera;
    ptz1 = anno(i).ptz;
    
    image_name2 = fullfile(base_dir, anno(i+1).image_name);
    camera2 = anno(i+1).camera;
    ptz2 = anno(i+1).ptz;
    
    % extract sift feature of current frame and next frame
    % save feature of current frame   
    
    im1 = imread(image_name1);
    im2 = imread(image_name2);
    
    % Load an image
    %im1([y: y+ h], [x: x+ w],:) = mean(im1(:)); % TV score area
    %im2([y: y+ h], [x: x+ w], :) = mean(im2(:));
    % Compute SIFT features for each
   
    [frames1, descrs1] = getFeatures(im1, 'peakThreshold', 0.01) ;
    [frames2, descrs2] = getFeatures(im2, 'peakThreshold', 0.01) ;
    
    [matches_geo, H] = getInliers(frames1, descrs1, frames2, descrs2);
    % experimentally set a magic number
    t = abs(sum(H(:)));
    if t > 2000
        continue;
    end

    figure(1) ; clf ;
    set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification') ;
    plotMatches(im1,im2,frames1,frames2,matches_geo, 'homography', H) ;
    title('Matches filtered by geometric verification');
    image_name1
    pause(1);

    % first image
    % filter outlier
    inlier = matches_geo(1,:);
    keypoint = frames1(:, inlier);
    descriptor = descrs1(:, inlier);

    % get (x, y) location
    keypoint = double(keypoint([1:2], :));
    keypoint = keypoint';
    descriptor = double(descriptor);
    descriptor = normc(descriptor);
    descriptor = descriptor';

    [pathstr,name,ext] = fileparts(image_name1);
    mat_name = strcat(name, '_1_soccer1_seq2.mat');
    im_name = strcat(name, ext);
    camera = camera1';
    ptz = ptz1;
    %save(mat_name, 'im_name', 'camera', 'ptz', 'keypoint', 'descriptor');

    % second image
    % filter outlier
    inlier = matches_geo(2,:);
    keypoint = frames2(:, inlier);
    descriptor = descrs2(:, inlier);

    % get (x, y) location
    keypoint = double(keypoint([1:2], :));
    keypoint = keypoint';
    descriptor = double(descriptor);
    descriptor = normc(descriptor);
    descriptor = descriptor';

    [pathstr,name,ext] = fileparts(image_name2);
    mat_name = strcat(name, '_2_soccer1_seq2.mat');
    im_name = strcat(name, ext);
    camera = camera2';
    ptz = ptz2;
    save(mat_name, 'im_name', 'camera', 'ptz', 'keypoint', 'descriptor');    
end
