
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <unordered_map>
#include <set>
#include <stdio.h>

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, 
                     float minimumReflectiveness, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            if ((*it2).r < minimumReflectiveness)
                continue;
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(std::vector<BoundingBox> &boundingBoxes,
                              float shrinkFactor,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches)
{
    // current = train
    // previous = query

    for (auto & match : kptMatches)
    {
        cv::KeyPoint &prevkp = kptsPrev[match.queryIdx];
        cv::KeyPoint &currkp = kptsCurr[match.trainIdx];

        vector<vector<BoundingBox>::iterator> enclosingBoxes;
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            auto &boundingBox = *it2;

            cv::Rect smallerBox;
            smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
            smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
            smallerBox.width =  boundingBox.roi.width * (1 - shrinkFactor);
            smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);

            if (smallerBox.contains(prevkp.pt) && smallerBox.contains(currkp.pt))
            {
                enclosingBoxes.push_back(it2);
            }
        }

        if (enclosingBoxes.size() == 1)
        {
            enclosingBoxes[0]->kptMatches.push_back(match);
        }
    }

    for (auto& bb : boundingBoxes)
    {
        std::vector<cv::DMatch> final_matches;
        std::vector<double> eucleadian_dist;
        for (auto& match : bb.kptMatches)
        {
            cv::KeyPoint &prevkp = kptsPrev[match.queryIdx];
            cv::KeyPoint &currkp = kptsCurr[match.trainIdx];

            eucleadian_dist.push_back(
                std::sqrt(
                    ((prevkp.pt.x - currkp.pt.x) * (prevkp.pt.x - currkp.pt.x)) +
                    ((prevkp.pt.y - currkp.pt.y) * (prevkp.pt.y - currkp.pt.y))
                )
            );
        }

        double mean = std::accumulate(eucleadian_dist.begin(), eucleadian_dist.end(), 0.0)/eucleadian_dist.size();
        auto add_square = [mean](double current_accumulation, double  elem)
        {
            auto d = elem - mean;
            return current_accumulation + d*d;
        };
        double sigmaN = std::accumulate(eucleadian_dist.begin(), eucleadian_dist.end(), 0.0, add_square);
        double standard_deviation = std::sqrt( sigmaN / eucleadian_dist.size());

        // just use the matches that their distance are within the standard deviation
        for (size_t index = 0; index < bb.kptMatches.size(); index++)
        {
            if (std::fabs( eucleadian_dist[index] - mean) < standard_deviation )
                final_matches.push_back(bb.kptMatches[index]);
        }
        bb.kptMatches = final_matches;
    }
    // std:: cout << temp_result.size() - boundingBox.kptMatches.size() << " removed" << std::endl;

}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    double minDist = 100.0; // min. required distance
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        volatile cv::DMatch ma = (*it1);
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    

    float dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate,
                     double &TTC, 
                     float minimumReflectiveness)
{
    double minXPrev = 1e9, minXCurr = 1e9;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->r > minimumReflectiveness)
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->r > minimumReflectiveness)
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr / frameRate / (minXPrev - minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame,
                        DataFrame &currFrame)
{

    // current = train
    // previous = query

    // create a map of bounding box pair -> number of matching occurrence
    std::unordered_map<bb_pairs, int, CustomHash> occurrence_map;

    for (auto &match : matches)
    {
        // which bounding boxes have the keypoint in the **previous** frame
        cv::KeyPoint &prev_kpt = prevFrame.keypoints[match.queryIdx];
        std::vector<int> prev_bb_containing_this_keypoint = findBoundingBoxesContainingKeypoint(prev_kpt, prevFrame);

        // which bounding boxes have the keypoint in the **current** frame
        cv::KeyPoint &current_kpt = currFrame.keypoints[match.trainIdx];
        std::vector<int> cur_bb_containing_this_keypoint = findBoundingBoxesContainingKeypoint(current_kpt, currFrame);

        // increment the occurrence map for all enclosing bounding boxes.
        for (int &prev_id : prev_bb_containing_this_keypoint)
        {
            for (int &cur_id : cur_bb_containing_this_keypoint)
            {
                occurrence_map[{prev_id, cur_id}] += 1;
            }
        }
    }

    // Declaring a set that will store the pairs using above comparision logic
    // hints from https://thispointer.com/how-to-sort-a-map-by-value-in-c/
    std::set<std::pair<bb_pairs, int>, comp__f> boundingBoxPairs(
        occurrence_map.begin(), occurrence_map.end(), comp__f());

    // for debug purposes print the set
    bool debugSorting = false;
    if (debugSorting)
    {
        for (std::pair<bb_pairs, int> bbmatch : boundingBoxPairs)
        {
            printf("match: { %d , %d }, occurred: %d \n", bbmatch.first.first, bbmatch.first.second, bbmatch.second);
        }
    }

    //create the final results
    bbBestMatches.clear();
    std::set<int> visited_BB_prev, visited_BB_current;
    for (std::pair<bb_pairs, int> bbmatch : boundingBoxPairs)
    {
        int bb_id_prev = bbmatch.first.first;
        int bb_id_cur = bbmatch.first.second;
        if (visited_BB_prev.find(bb_id_prev) == visited_BB_prev.end() &&
            visited_BB_current.find(bb_id_cur) == visited_BB_current.end())
        {
            visited_BB_prev.insert(bb_id_prev);
            visited_BB_current.insert(bb_id_cur);
            bbBestMatches.emplace(std::make_pair(bb_id_prev, bb_id_cur));
        }
    }

    if (false)
    {
        printf(" %s: number of matched bbs: %lu \n", __FUNCTION__, bbBestMatches.size());
    }
}

std::vector<int> findBoundingBoxesContainingKeypoint(cv::KeyPoint kpt, DataFrame frame)
{
    std::vector<int> res;
    for (auto &bb : frame.boundingBoxes)
    {
        if (bb.contains(kpt))
        {
            res.push_back(bb.boxID);
        }
    }
    return res;
}