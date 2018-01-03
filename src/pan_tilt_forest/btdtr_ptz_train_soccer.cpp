//
//  btdtr_ptz_train.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-07-28.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//
#if 1
#include <stdio.h>
#include <iostream>
#include "btdtr_ptz_builder.h"
#include "bt_dt_regressor.h"
#include "mat_io.hpp"
#include "cvx_util.hpp"

using Eigen::MatrixXf;
using Eigen::MatrixXi;

static void help()
{
    printf("program          trainDataFolder  BTDTParamFile saveFile\n");
    printf("BTDTR_PTZ_train  data/.mat        param.txt     dt_model.txt\n");
    printf("trainDataFolder: a folder of .mat files. Each .mat file has ptz, keypoint, descriptor \n");
    printf("BTDTParamFile: back tracking decision tree parameters\n");
    printf("saveFile: learned decision tree model \n");
}


int main(int argc, const char * argv[])
{
    if (argc != 4) {
        printf("argc is %d, should be 4.\n", argc);
        help();
        return -1;
    }
    
    const char * seq_file = argv[1];
    const char * tree_param_file = argv[2];
    const char * model_file = argv[3];
    
    /*
    const char * seq_file = "/Users/jimmy/Desktop/images/bmvc17_soccer/random_forest/ptz_sift_inlier/*.mat";
    const char * tree_param_file = "/Users/jimmy/Desktop/BTDT_ptz/ptz_tree_param_debug.txt";
    const char * model_file = "test_model.txt";
     */
    
    vector<string> feature_files;
    CvxUtil::readFilenames(seq_file, feature_files);
    
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(tree_param_file);
    tree_param.printSelf();
    
    BTDTRPTZBuilder builder;
    builder.setTreeParameter(tree_param);
    
    BTDTRegressor model;
    builder.buildModel(model, feature_files, model_file);
    model.saveModel(model_file);
    
    return 0;
}
#endif
