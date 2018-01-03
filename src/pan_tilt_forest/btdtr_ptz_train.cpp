//
//  btdtr_ptz_train.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-07-28.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#if 0
#include <stdio.h>
#include <iostream>
#include "btdtr_ptz_builder.h"
#include "bt_dt_regressor.h"
#include "mat_io.hpp"

using Eigen::MatrixXf;
using Eigen::MatrixXi;

static void help()
{
    printf("program          seqFile       baseDir   BTDTParamFile saveFile\n");
    printf("BTDTR_PTZ_train  feat_seq1.txt ./seq/    param.txt     dt_model.txt\n");
    printf("seqFile: a text file has a sequence of (feature, label) pairs . \n");
    printf("baseDir: base directory of the sequence of features. Features are stored in .mat file. \n");
    printf("Basketball 4 sequence: /Users/jimmy/Desktop/images/Youtube/PTZRegressor/four_sequences/ \n");
    printf("BTDTParamFile: back tracking decision tree parameters\n");
    printf("saveFile: learned decision tree model \n");
}


int main(int argc, const char * argv[])
{    
    if (argc != 5) {
        printf("argc is %d, should be 5.\n", argc);
        help();
        return -1;
    }
    
    const char * seq_file = argv[1];
    const char * seq_base_directory = argv[2];
    const char * tree_param_file = argv[3];
    const char * model_file = argv[4];
    
    
    /*
    const char * seq_file = "/Users/jimmy/Desktop/BTDT_ptz/data/inlier_feat_seq1.txt";
    const char * seq_base_directory = "/Users/jimmy/Desktop/images/Youtube/PTZRegressor/four_sequences/";
    const char * tree_param_file = "/Users/jimmy/Desktop/BTDT_ptz/ptz_tree_param_debug.txt";
    const char * model_file = "test_model.txt";
     */
    
    
    vector<string> feature_files;
    vector<Eigen::Vector3f> ptzs;
    btdtr_ptz_util::readSequenceData(seq_file, seq_base_directory, feature_files, ptzs);    
    
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(tree_param_file);
    tree_param.printSelf();
    
    BTDTRPTZBuilder builder;
    builder.setTreeParameter(tree_param);
    
    BTDTRegressor model;
    builder.buildModel(model, feature_files, ptzs, model_file);
    model.saveModel(model_file);
    
    return 0;
}
#endif
