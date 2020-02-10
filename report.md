# Report

## Content
- [Introduction](#Introduction)
- [FP1 - Match 3D Objects](#FP1)
- [FP2 - Compute Lidar-based TTC](#FP2)
- [FP3 - Associate Keypoint Correspondences with Bounding Boxes](#FP3)
- [FP4 - Compute Camera-based TTC](#FP4)
- [FP5 - Performance Evaluation 1](#FP5)
- [FP6 - Performance Evaluation 2](#FP6)

<a name="Introduction" />

## Introduction
This is a report to cover the PROJECT SPECIFICATION for 3rd project of Sensor Fusion Nanodegree, **3D Object Tracking**.


<a name="FP1" />

## FP1 - Match 3D Objects

The goal in this section is to match bounding boxes between two frames. In order to 
achieve this, a map from bounding pairs to the number of matched keypoints in those 
frames is created which is defined like the following in the `matchBoundingBoxes` 
function.
```c++
typedef std::pair<int, int> bb_pairs; // pair of bounding box ids
std::unordered_map<bb_pairs, int, CustomHash> occurrence_map;
```
Then all the matches are iterated and `occurrence_map` is incremented for any keypoint
match that has been visited inside the `matches` vector. Then the `occurrence_map` 
is sorted based on the number of occurrence in a set as the following:
```c++
struct comp__f
{
    bool operator()(const std::pair<bb_pairs, int> &lhs, const std::pair<bb_pairs, int> &rhs) const
    {
        return lhs.second > rhs.second;
    }
};

std::set<std::pair<bb_pairs, int>, comp__f> boundingBoxPairs(
        occurrence_map.begin(), occurrence_map.end(), comp__f());
```
and then this sorted set is used to match the bounding boxes. Complete code can be found in the `camFusion_Student.cpp` file.

<a name="FP2" />

## FP2 - Compute Lidar-based TTC

The equation to calculate the time to collision (TTC) to the proceeding vehicle that is used for lidar is as follow:

<img align="middle" src="https://latex.codecogs.com/gif.latex?TTC&space;=&space;\frac{d_1&space;\Delta&space;t}{d_0&space;-&space;d_1}," title="TTC = \frac{d_1 \Delta t}{d_0 - d_1}," />

which <img src="https://latex.codecogs.com/gif.latex?d_0" title="d_0" /> and 
<img src="https://latex.codecogs.com/gif.latex?d_1" title="d_1" /> are the distances to 
the proceeding vehicle in two different consecutive frame (timestamp) and 
<img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" /> is the 
time difference between these two measurements.

Different approach can be used to find the distance to the proceeding vehicle. 
First the points that have reflection less than a threshold are filtered out and then 
distance to the closest point is considered as the distance to the proceeding vehicle 
in each frame. Calculation of TTC is is implemented as the following

```c++
#define REFLECTION_THRESHOLD 0.2

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate,
                     double &TTC)
{
    double minXPrev = 1e9, minXCurr = 1e9;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->r > REFLECTION_THRESHOLD)
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->r > REFLECTION_THRESHOLD)
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr / frameRate / (minXPrev - minXCurr);
}
```

<a name="FP3" />

## FP3 - Associate Keypoint Correspondences with Bounding Boxes
In this section, the goal is to find te keypoints (matches) that belongs to a specific 
bounding box. In order to do that, first all the matches are iterated to be checked if pair
of keypoints in the match are inside the bounding box. The distance between all the pairs 
are also calculated which will be used to filter out erroneous matches.

```c++
std::vector<cv::DMatch> temp_result;
std::vector<double> eucleadian_distances;
for (auto &match : kptMatches)
{
    cv::KeyPoint &prevkp = kptsPrev[match.queryIdx];
    cv::KeyPoint &currkp = kptsCurr[match.trainIdx];
    if (boundingBox.contains(prevkp) && boundingBox.contains(currkp))
    {
        temp_result.push_back(match);
        eucleadian_distances.push_back(
            std::sqrt(
                ((prevkp.pt.x - currkp.pt.x) * (prevkp.pt.x - currkp.pt.x)) +
                ((prevkp.pt.y - currkp.pt.y) * (prevkp.pt.y - currkp.pt.y))
            )
        );
    }
}
```

Then the mean and standard deviation of all the distances are calculated and the pairs which 
are too far away (more than half of the standard deviation) are filtered out.

```c++
// calculate the mean and std of distances
double mean = std::accumulate(eucleadian_distances.begin(), eucleadian_distances.end(), 0.0)/eucleadian_distances.size();
auto add_square = [mean](double current_accumulation, double  elem)
{
    auto d = elem - mean;
    return current_accumulation + d*d;
};
double sigmaN = std::accumulate(eucleadian_distances.begin(), eucleadian_distances.end(), 0.0, add_square);
double standard_deviation = std::sqrt( sigmaN / eucleadian_distances.size());
// just use the matches that their distance are within the standard deviation
for (size_t index = 0; index < temp_result.size(); index++)
{
    if (std::fabs( eucleadian_distances[index] - mean) < standard_deviation/2 )
        boundingBox.kptMatches.push_back(temp_result[index]);
}
```

The complete code can be found in `clusterKptMatchesWithROI` function of 
`camFusion_Student.cpp` file.

<a name="FP4" />

## FP4 - Compute Camera-based TTC

<img align="middle" src="https://latex.codecogs.com/gif.latex?TTC&space;=&space;\frac{-\Delta&space;t}{1-\frac{h_1}{h_0}}" title="TTC = \frac{-\Delta t}{1-\frac{h_1}{h_0}}" />

<a name="FP5" />

## FP5 - Performance Evaluation 1

<a name="FP6" />

## FP6 - Performance Evaluation 2
