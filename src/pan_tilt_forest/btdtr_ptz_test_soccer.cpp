//
//  btdtr_ptz_test.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-07-28.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0

#include <stdio.h>
#include <iostream>
#include <time.h>
#include "btdtr_ptz_builder.h"
#include "bt_dt_regressor.h"
#include "mat_io.hpp"
#include <iostream>
#include "ptz_pose_estimation.h"
#include "cvx_util.hpp"
#include "dt_util.hpp"

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using namespace std;

static void help()
{
    printf("program        modelFile   ptz_feature_folder  maxCheck reprojThreshold distanceThreshold sampleNumber maxTreeNum saveFile  \n");
    printf("BTDTR_PTZ_test model.txt   ptz_sift/*.mat      4        2               0.2               32           -1         result.mat \n");
    printf("modelFile: a random forest . \n");
    printf("ptz_feature_folder: .mat file contains ptz, keypoint and descriptor.\n");
    printf("maxCheck: back tracking decision tree parameter, the larger the slower\n");
    printf("reprojThreshold: camera pose estimation parameter. unit pixel\n");
    printf("distanceThreshold: (SIFT) feature distance threhold \n");
    printf("sampleNumber: preemptive RANSAC sample number in each iteration \n");
    printf("maxTreeNum: number of trees used in the test. For ablation experiment. -1 for use all trees.\n");
    printf("saveFile: .mat file ground truth PTZ and estimated PTZ \n");
    printf("default image size is 1280 x 720\n");
    
}


int main(int argc, const char * argv[])
{
    if (argc != 9) {
        printf("argc is %d, should be 9.\n", argc);
        help();
        return -1;
    }
    
    const char * model_file = argv[1];
    const char * ptz_feature_folder = argv[2];
    const int max_check = strtod(argv[3], NULL);
    const double reprojection_error_threshold = strtod(argv[4], NULL);
    const double distance_threshold = strtod(argv[5], NULL);
    const int sample_number = (int)strtod(argv[6], NULL);
    int maxTreeNum = (int)strtod(argv[7], NULL);
    const char * save_file = argv[8];
    
    /*
    const char * model_file = "/Users/jimmy/Desktop/BTDT_ptz_soccer/model/seq2_model.txt";
    const char * ptz_feature_folder = "/Users/jimmy/Desktop/BTDT_ptz_soccer/soccer_data/train_data/seq1_ptz_sift_inlier/*.mat";
    const int max_check =  8;
    const double reprojection_error_threshold = 2.0;
    const double distance_threshold = 0.2;
    const int sample_number = 32;
    const char * save_file = "result.mat";
     */
    
    // read model
    BTDTRegressor model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    if (maxTreeNum == -1) {
        maxTreeNum = model.treeNum();
    }
    printf("use %d trees in the test\n", maxTreeNum);
    
    // read testing examples
    vector<string> feature_files;
    CvxUtil::readFilenames(ptz_feature_folder, feature_files);
    
    Eigen::Vector2d pp(1280.0/2.0, 720.0/2.0);
    ptz_pose_opt::PTZPreemptiveRANSACParameter param;
    param.reprojection_error_threshold_ = reprojection_error_threshold;
    param.sample_number_ = sample_number;
    printf("principal point is fixed at %lf %lf\n", pp.x(), pp.y());
    printf("inlier reprojection error is %lf pixels\n", param.reprojection_error_threshold_);
    
    Eigen::MatrixXd gt_ptz_all(feature_files.size(), 3);
    Eigen::MatrixXd estimated_ptz_all(feature_files.size(), 3);
    int index = 0;
    for (const string &file_name: feature_files) {
        vector<btdtr_ptz_util::PTZSample> samples;
        Eigen::Vector3f ptz;
        btdtr_ptz_util::generatePTZSampleWithFeature(file_name.c_str(), pp.cast<float>(), ptz, samples);
        Eigen::Vector3d ptz_gt(ptz.x(), ptz.y(), ptz.z());
        printf("feature number is %lu\n", samples.size());
        
        vector<Eigen::Vector2d> image_points;
        vector<vector<Eigen::Vector2d> > candidate_pan_tilt;
        Eigen::Vector3d estimated_ptz(0, 0, 0);
        
        // predict from observation (descriptors)
        double tt = clock();
        for (int j = 0; j<samples.size(); j++) {
            btdtr_ptz_util::PTZSample s = samples[j];
            Eigen::VectorXf feat = s.descriptor_;
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> cur_dists;
            model.predict(feat, max_check, maxTreeNum, cur_predictions, cur_dists);
            assert(cur_predictions.size() == cur_dists.size());
            
            //cout<<"minimum feature distance "<<cur_dists[0]<<endl;
            if (cur_dists[0] < distance_threshold) {
                image_points.push_back(Eigen::Vector2d(s.loc_.x(), s.loc_.y()));
                vector<Eigen::Vector2d> cur_candidate;
                for (int k = 0; k<cur_predictions.size(); k++) {
                    assert(cur_predictions[k].size() == 2);
                    if (cur_dists[k] < distance_threshold) {
                        cur_candidate.push_back(Eigen::Vector2d(cur_predictions[k][0], cur_predictions[k][1]));
                    }
                }
                candidate_pan_tilt.push_back(cur_candidate);
            }
        }
        // estimate camera pose
        bool is_opt = ptz_pose_opt::preemptiveRANSACOneToMany(image_points, candidate_pan_tilt, pp,
                                                param, estimated_ptz, false);
        printf("Prediction and camera pose estimation cost time: %f seconds.\n", (clock() - tt)/CLOCKS_PER_SEC);
        if (is_opt) {
            cout<<"ptz estimation error: "<<(ptz_gt - estimated_ptz).transpose()<<endl;
        }
        else {
            printf("-------------------------------------------- Optimize PTZ failed.\n");
            printf("valid feature number is %lu\n\n", image_points.size());
        }
        gt_ptz_all.row(index) = ptz_gt;
        estimated_ptz_all.row(index) = estimated_ptz;
        index++;
    }
    assert(index == gt_ptz_all.rows());
    
    std::vector<std::string> var_name;
    std::vector<Eigen::MatrixXd> data;
    var_name.push_back("ground_truth_ptz");
    var_name.push_back("estimated_ptz");
    data.push_back(gt_ptz_all);
    data.push_back(estimated_ptz_all);
    matio::writeMultipleMatrix(save_file, var_name, data);    
    
    // simple statistics
    // void meanMedianError(const vector<T> & errors, T & mean, T & median);
    vector<Eigen::VectorXd> errors;
    for (int r = 0; r<gt_ptz_all.rows(); r++) {
        Eigen::VectorXd err = estimated_ptz_all.row(r) - gt_ptz_all.row(r);
        errors.push_back(err);
    }
    
    Eigen::VectorXd mean_error, median_error;
    dt::meanMedianError(errors, mean_error, median_error);
    cout<<"PTZ mean   error: "<<mean_error.transpose()<<endl;
    cout<<"PTZ median error: "<<median_error.transpose()<<endl;
    
    return 0;
}
#endif
