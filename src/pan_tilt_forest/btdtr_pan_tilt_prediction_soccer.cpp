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
    printf("program        modelFile   ptz_feature_folder  maxCheck distanceThreshold saveFolder  \n");
    printf("BTDTR_PTZ_test model.txt   ptz_sift/*.mat      4        0.2               ./result \n");
    printf("Pan, tilt prediction error measurement.\n");
    printf("modelFile: a random forest . \n");
    printf("ptz_feature_folder: .mat file contains ptz, keypoint and descriptor.\n");
    printf("maxCheck: back tracking decision tree parameter, the larger the slower\n");
    printf("distanceThreshold: (SIFT) feature distance threhold \n");
    printf("saveFolder: a folder to save all predicted pan and tilt \n");
    printf("default image size is 1280 x 720\n");
}


int main(int argc, const char * argv[])
{
    if (argc != 6) {
        printf("argc is %d, should be 6.\n", argc);
        help();
        return -1;
    }
     
    const char * model_file = argv[1];
    const char * ptz_feature_folder = argv[2];
    const int max_check = strtod(argv[3], NULL);
    const double distance_threshold = strtod(argv[4], NULL);
    const char * save_folder = argv[5];
    
    /*
    const char * model_file = "/Users/jimmy/Desktop/BTDT_ptz_soccer/model/leave_1_out.txt";
    const char * ptz_feature_folder = "/Users/jimmy/Desktop/BTDT_ptz_soccer/soccer_data/test_data/seq1_raw/*.mat";
    const int max_check =  4;
    const double distance_threshold = 0.2;
    const char * save_folder = "./result";
     */
    
    // read model
    BTDTRegressor model;
    bool is_read = model.load(model_file);
    assert(is_read);
    
    // read testing examples
    vector<string> feature_files;
    CvxUtil::readFilenames(ptz_feature_folder, feature_files);
    
    Eigen::Vector2f pp(1280.0/2.0, 720.0/2.0);
    printf("principal point is fixed at %lf %lf\n", pp.x(), pp.y());
    
    for (const string &file_name: feature_files) {
        vector<btdtr_ptz_util::PTZSample> samples;
        Eigen::Vector3f ptz;
        btdtr_ptz_util::generatePTZSampleWithFeature(file_name.c_str(), pp, ptz, samples);
        Eigen::Vector3d ptz_gt(ptz.x(), ptz.y(), ptz.z());
        printf("feature number is %lu\n", samples.size());
        
        // predict from observation (descriptors)
        Eigen::MatrixXd gt_pan_tilt_all(samples.size(), 2);
        Eigen::MatrixXd estimated_pan_tilt_all(samples.size(), 2);
        int inlier_num = 0;
        for (int j = 0; j<samples.size(); j++) {
            btdtr_ptz_util::PTZSample s = samples[j];
            Eigen::VectorXf feat = s.descriptor_;
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> cur_dists;
            model.predict(feat, max_check, cur_predictions, cur_dists);
            assert(cur_predictions.size() == cur_dists.size());
            
            if (cur_dists[0] < distance_threshold) {
                gt_pan_tilt_all(inlier_num, 0) = s.pan_tilt_[0];
                gt_pan_tilt_all(inlier_num, 1) = s.pan_tilt_[1];
                estimated_pan_tilt_all(inlier_num, 0) = cur_predictions[0][0];
                estimated_pan_tilt_all(inlier_num, 1) = cur_predictions[0][1];
                inlier_num++;
            }
        }
        if (inlier_num > 0) {
            gt_pan_tilt_all = gt_pan_tilt_all.topRows(inlier_num);
            estimated_pan_tilt_all = estimated_pan_tilt_all.topRows(inlier_num);
            // save file
            string path, cur_file_name;
            CvxUtil::splitFilename(file_name, path, cur_file_name);
            cur_file_name = string(save_folder) + string("/pan_tilt_") + cur_file_name;
            
            std::vector<std::string> var_name;
            std::vector<Eigen::MatrixXd> data;
            var_name.push_back("gt_pan_tilt");
            var_name.push_back("estimated_pan_tilt");
            data.push_back(gt_pan_tilt_all);
            data.push_back(estimated_pan_tilt_all);
            matio::writeMultipleMatrix(cur_file_name.c_str(), var_name, data);
        }
        else {
            printf("Warning: prediction failed, no inlier prediction.\n");
        }
    }
    
    return 0;
}
#endif
