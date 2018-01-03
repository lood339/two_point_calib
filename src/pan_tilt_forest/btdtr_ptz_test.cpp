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
#include "btdtr_ptz_builder.h"
#include "bt_dt_regressor.h"
#include "mat_io.hpp"
#include <iostream>

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using namespace std;

static void help()
{
    printf("program        modelFile   seqFile       baseDir   maxCheck saveDir\n");
    printf("BTDTR_PTZ_test model.txt   feat_seq1.txt ./seq/    8        result\n");
    printf("modelFile: a random forest . \n");
    printf("seqFile: a text file has a sequence of (feature, label) pairs. Label is not acutally used.\n");
    printf("baseDir: base directory of the sequence of features. Features are stored in .mat file. \n");
    printf("Basketball 4 sequence: /Users/jimmy/Desktop/images/Youtube/PTZRegressor/four_sequences/ \n");
    printf("maxCheck: back tracking decision tree parameter, the larger the slower\n");
    printf("saveDir: folder to save predictions \n");
    printf("default image size is 1280 x 720\n");
}

static Eigen::VectorXf toLongVector(const vector<Eigen::VectorXf> & features)
{
    long dim = features[0].size();
    Eigen::VectorXf feat(dim * features.size());
    
    for (int i = 0; i<features.size(); i++) {
        for (int j = 0; j<features[i].size(); j++) {
            feat[i * dim + j] = features[i][j];
        }
    }
    return feat;
}

static Eigen::MatrixXf toMatrix(const vector<Eigen::VectorXf>& features)
{
    Eigen::MatrixXf mat = Eigen::MatrixXf::Zero(features.size(), features[0].size());
    for (int i = 0; i<features.size(); i++) {
        mat.row(i) = features[i];
    }
    return mat;
}

static Eigen::VectorXf toLongVector(const vector<float> & dist)
{
    Eigen::VectorXf feat(dist.size());
    for (int i = 0; i<dist.size(); i++) {
        feat[i] = dist[i];
    }
    return feat;
}

int main(int argc, const char * argv[])
{
    if (argc != 6) {
        printf("argc is %d, should be 6.\n", argc);
        help();
        return -1;
    }
    
    const char * model_file = argv[1];
    const char * seq_file = argv[2];
    const char * seq_base_directory = argv[3];
    const int max_check = strtod(argv[4], NULL);
    const char * save_dir = argv[5];
    
    /*
    const char * model_file = "/Users/jimmy/Desktop/BTDT_ptz/model/debug.txt";
    const char * seq_file = "/Users/jimmy/Desktop/BTDT_ptz/data/inlier_feat_seq1.txt";
    const char * seq_base_directory = "/Users/jimmy/Desktop/images/Youtube/PTZRegressor/four_sequences/";
    const int max_check =  8;
    const char * save_dir = "result";
     */
    
    BTDTRegressor model;
    model.load(model_file);
    
    vector<string> feature_files;
    vector<Eigen::Vector3f> ptzs;
    btdtr_ptz_util::readSequenceData(seq_file, seq_base_directory, feature_files, ptzs);
    
    Eigen::Vector2f pp(640, 360);
    cout<<"principal point "<<pp.transpose()<<endl;
    
    for (int i = 0; i<feature_files.size(); i++) {
        vector<btdtr_ptz_util::PTZSample> samples;
        btdtr_ptz_util::generatePTZSample(feature_files[i].c_str(), pp, ptzs[i], samples);
        
        vector<Eigen::VectorXf> locations;
        vector<Eigen::VectorXf> labels;
        
        vector<Eigen::VectorXf> predictions;
        vector<Eigen::VectorXf> distances;
        
        for (int j = 0; j<samples.size(); j++) {
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> cur_dists;
            model.predict(samples[j].descriptor_, max_check, cur_predictions, cur_dists);
            labels.push_back(samples[j].pan_tilt_);
            locations.push_back(samples[j].loc_);
            
            // group predictions and distances
            predictions.push_back(toLongVector(cur_predictions));
            distances.push_back(toLongVector(cur_dists));
        }
        
        vector<string> var_names;
        vector<Eigen::MatrixXf> data;
        var_names.push_back("location");
        var_names.push_back("label");
        var_names.push_back("prediction");
        var_names.push_back("distance");
        data.push_back(toMatrix(locations));
        data.push_back(toMatrix(labels));
        data.push_back(toMatrix(predictions));
        data.push_back(toMatrix(distances));
        
        char buf[1024] = {NULL};
        sprintf(buf, "%s/%08d.mat", save_dir, i);
        matio::writeMultipleMatrix<Eigen::MatrixXf>(buf, var_names, data);
    }    
    
    return 0;
}
#endif
