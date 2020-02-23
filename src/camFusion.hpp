
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"

void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT);
void clusterKptMatchesWithROI(std::vector<BoundingBox> &boundingBoxes,
                              float shrinkFactor,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame,
                        DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes,
                   cv::Size worldSize,
                   cv::Size imageSize, 
                   float minimumReflectiveness,
                   bool bWait = true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches,
                      double frameRate,
                      double &TTC,
                      cv::Mat *visImg = nullptr);

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate,
                     double &TTC, 
                     float minimumReflectiveness);

std::vector<int> findBoundingBoxesContainingKeypoint(cv::KeyPoint kpt,
                                                               DataFrame frame);


#endif /* camFusion_hpp */
