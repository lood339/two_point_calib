//
//  btdtr_ptz_builder.h
//  PTZBTRF
//
//  Created by jimmy on 2017-07-20.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__btdtr_ptz_builder__
#define __PTZBTRF__btdtr_ptz_builder__

#include <stdio.h>
#include <Eigen/Dense>
#include <string>
#include "bt_dt_regressor.h"
#include "btdtr_ptz_util.h"

using std::string;
class BTDTRPTZBuilder {

    using TreeParameter = btdtr_ptz_util::PTZTreeParameter;
    using TreeType = BTDTRTree;
    typedef TreeType* TreePtr;    

    TreeParameter tree_param_;
    
public:

    BTDTRPTZBuilder();
    ~BTDTRPTZBuilder();

    void setTreeParameter(const TreeParameter& param);

    // build model from subset of images
    // sift feature are precomputed to save time
    // feature_files: has keypoint location and feature descriptor
    bool buildModel(BTDTRegressor& model,                    
                    const vector<string> & feature_files,
                    const vector<Eigen::Vector3f> & ptzs,
                    const char *model_file_name) const;
    
    // ptz_keypoint_descriptor_files: .mat file has ptz, keypoint location and descriptor    //
    bool buildModel(BTDTRegressor& model,
                    const vector<string> & ptz_keypoint_descriptor_files,
                    const char *model_file_name) const;
    
private:
    
    bool validationError(const BTDTRegressor & model,
                          const vector<string> & feature_files,
                          const vector<Eigen::Vector3f> & ptzs,
                          const int sample_frame_num = 10) const;
    
    bool validationError(const BTDTRegressor & model,
                         const vector<string> & ptz_keypoint_descriptor_files,
                         const int sample_frame_num = 10) const;



};

#endif /* defined(__PTZBTRF__btdtr_ptz_builder__) */
