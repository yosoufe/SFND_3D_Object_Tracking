# SFND 3D Object Tracking

This project practices the following:

1. Understanding of keypoint detectors, descriptors, and methods to match them between successive images.
2. How to detect objects in an image using the YOLO deep-learning framework.
3. How to associate regions in a camera image with Lidar points in 3D space.

Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, the missing parts in the schematic are being implemented in four major tasks: 
1. First, a way to match 3D objects over time by using keypoint correspondences is developed. 
2. Second, computing the TTC based on Lidar measurements is implemented. 
3. and then computing TTC using camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, various tests with the framework are being conducted. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* for argument parsing I am using [argparse in this github repo](https://github.com/cofyc/argparse)
   which is already included as a submodule here.

## Basic Build Instructions

1. Clone this repo and submodule [argparse](https://github.com/cofyc/argparse) as follow.

   ```
   git clone https://github.com/yosoufe/SFND_2D_Feature_Tracking.git
   cd SFND_2D_Feature_Tracking
   git submodule update --init --recursive
   ```

2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile without GPU Support: `cmake .. && make`

      * With GPU support: `cmake -DWITH_CUDA=ON .. && make`:
         
         In case with of GPU support the extra following items would be available: 
            
          * detectors: **ORB_CUDA** and **FAST_CUDA**
          * matcher: **MAT_BF_CUDA**
          * descriptor: **ORB_CUDA**

4. Run it:

   * The executable accepts multiple optional arguments to define multiple variables.
   * use `./3D_object_tracking -h` for help. It would create the following output:

        ```
        $ ./3D_object_tracking -h
        Usage: ./3D_feature_tracking [args]
        For example: ./3D_feature_tracking --detector_type=BRISK --matcher_type=MAT_FLANN --descriptor_type=DES_BINARY --selector_type=SEL_KNN

        Explores different 2d keypoint detector, descriptor and matching

            -h, --help                show this help message and exit

        Optional Arguments: 
            --detector_type=<str>     detector type, options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT
                                        if compiled (WITH_CUDA on): ORB_CUDA, FAST_CUDA
                                        default: ORB
            --matcher_type=<str>      matcher type, options: MAT_BF, MAT_FLANN,
                                        if compiled (WITH_CUDA on): MAT_BF_CUDA
                                        default: MAT_BF
            --descriptor_type=<str>   descriptor type, options: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
                                        if compiled (WITH_CUDA on): ORB_CUDA
                                        default: BRISK
            --selector_type=<str>     selector type, options: SEL_NN, SEL_KNN
                                        default: SEL_NN
            -f, --focus_on_vehicle    To focus on only keypoints that are on the preceding vehicle.
            -l, --limit_keypoints     To limit the number of keypoints to maximum 50 keypoints.
            --top_view                Lidar Top View
            --camera_view             Camera View
            -v, --verbose             logging the steps of the program that are being started or finished.
            -d, --debug               showing debug messages.
        ```