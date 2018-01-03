# two_point_calib
This is an implementation of "A Two-point Method for PTZ camera Calibration in Sports" (WACV2018)


Dependences:
1. OpenCV 3.1 or later. 
2. Eigen 3.2.6 or later.
3. flann 1.8.4 or later.
4. matio: https://github.com/tbeu/matio

The code is tested on Xcode 6.4 on a Mac 10.10.5 system. But the code has minimum dependence on compile and system, so it should work well on linux and windows as well.

File structure:
matlab: synthetic example of the two-point calibration method.

In the src/pan_tilt_forest folder
bt_dtr: a general implementation of regression forest with back tracking
 
cvx_gl and cvx_pgl: geometry and project geometry files

dt_util: decision tree utility

util: pan-tilt-zoom camera pose estimation given ray-3D correspondences


Todo: cmake file and training parameters
