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
The main goal is to calculate time to collision (TTC) to the proceeding vehicle 
using single camera and lidar.


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

This is alone is not giving a good results because still there would be a
lot of outliers like the following images:

<img src=results/FP2/NoFilter_TopView_01.png width=500>

<img src=results/FP2/NoFilter_CamView_01.png width=1000>

I am using 
[StatisticalOutlierRemoval](http://pointclouds.org/documentation/tutorials/statistical_outlier.php)
to remove the outliers. The result for the same frame would be the following images:

<img src=results/FP2/Filtered_TopView_01.png width=500>

<img src=results/FP2/Filtered_CamView_01.png width=1000>

Which the outliers are completely gone. This is integrated in `clusterLidarWithROI`
function as the following:

```c++
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);

    for (auto & p : lidarPoints)
    {
        pcl::PointXYZI pt;
        pt.x = p.x; pt.y = p.y; pt.z = p.z; pt.intensity = p.r;
        cloud->points.push_back(pt);
    }

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (20);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);

    std::vector<LidarPoint> filtered_lidar_points;
    for (auto&p : cloud_filtered->points)
    {
        LidarPoint pt;
        pt.x = p.x; pt.y = p.y; pt.z = p.z; pt.r = p.intensity;
        filtered_lidar_points.push_back(pt);
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

The complete code can be found in `clusterKptMatchesWithROI` function in 
`camFusion_Student.cpp` file.

<a name="FP4" />

## FP4 - Compute Camera-based TTC

Main equation to calculate the TTC based on monocular imaging is as the following:

<img align="middle" src="https://latex.codecogs.com/gif.latex?TTC&space;=&space;\frac{-\Delta&space;t}{1-\frac{h_1}{h_0}}" title="TTC = \frac{-\Delta t}{1-\frac{h_1}{h_0}}" />

which <img src="https://latex.codecogs.com/gif.latex?h_1" title="h_1" /> is 
the distance between two keypoints in the first frame and 
<img src="https://latex.codecogs.com/gif.latex?h_2" title="h_2" /> is
the distance between the same keypoints in the second frame, and
<img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" /> 
is the time difference between these frames. 
<img src="https://latex.codecogs.com/gif.latex?h_1/h_0" title="h_1/h_0" /> is called 
distance ration or `distRatio` in the code. Distance ration is calculated for all keypoint pairs which 
are at the minimum distance of 100 (pixels) from each other on the same frame. 

```c++
vector<double> distRatios; // stores the distance ratios for all keypoint
double minDist = 100.0; // min. required distance
for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
{ // outer kpt. loop
    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it
    { // inner kpt.-loop
        // get next keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
        cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
        // compute distances and distance ratios
        double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
        double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
        if (distPrev > std::numeric_limits<double>::epsilon() && distCurr
        { // avoid division by zero
            double distRatio = distCurr / distPrev;
            distRatios.push_back(distRatio);
        }
    } // eof inner loop over all matched kpts
}     // eof outer loop over all matched kpts
```

Then the median of the all distance ratios are used in the above equation
to calculate TTC as follow

```c++
// compute median dist. ratio to remove outlier influence
std::sort(distRatios.begin(), distRatios.end());
long medIndex = floor(distRatios.size() / 2.0);
double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];
    
float dT = 1 / frameRate;
TTC = -dT / (1 - medDistRatio);
```


I am also using the same technique here as in clustering the point clouds. Basically I am 
removing the keypoints that are belonging to two or more bounding boxes and only keep 
the ones that are unique to single bounding box. For this purpose I had to move the position
of this function to one layer on top and do minor changes to the prototype 
of the function `clusterKptMatchesWithROI`. For more details please take a look at the
source code.

<a name="FP5" />

## FP5 - Performance Evaluation 1

I have calculated the TTC with different minimum reflectiveness from the LIDAR data and made the 
following graph.

<img src=results/FP5/task_5.png width=1000>

The data for the above graph is saved in `results/lidar_ttcs_vs_ths.p` pickle file and 
in the jupyter notebook `analyse.ipynb` there is a small code snipper on how to use it.

I would continue the discussion on the following graph with `minimum reflectiveness = 0.2`.

<img src=results/FP5/TTCLidar0_2.png width=1000>

The first thing, that attracts attentions, is the measurements 
for the frames 52 and above. The question is 
why they are fluctuating so much. The answer is at those moments the preceding vehicle is 
standing still and TTC is very large and basically it is infinite. Therefore the calculated
number is jumping between a very big negative number and very big positive number.

On earlier frames there are some jumps visible that they do not seems to be real. 
For example at frames 37-38 if we check the top view of the lidar we would see the 
following images

<img src=results/FP5/noisy_37-38.png width=1000>

<img src=results/FP5/noisy_37-38_2.png width=1000>

As you can see there are very small noises, on the first image on right side
which causes that the car looks closer than real on the first image,
that makes the displacement between two frames less than real and causing 
to increase the TTC. The solution could be that we increase the resolution.
What I mean by increasing the resolution is that, instead of considering all
of the points as single box, we divide them into maybe 5 to 10 boxes (sticks as 
is described in the lectures) and calculate
the TTC for each box (stick) and use the average as final value.

The similar error can be seen on frames 29-30.

<img src=results/FP5/noisy_29-30.png width=1000>

<img src=results/FP5/noisy_29-30_2.png width=1000>

Because the noises are very small, the 
[StatisticalOutlierRemoval](http://pointclouds.org/documentation/tutorials/statistical_outlier.php)
filter could not eliminate them.

<a name="FP6" />

## FP6 - Performance Evaluation 2

I wrote the `runner.py` script to run the executable with different arguments 
on different CPU cores in parallel. So I ran it with all possible pairs of 
keypoint detectors and descriptors. The following graph is showing the results.

<img src=results/FP6/task_6.png width=1000>

The same story as the lidar graph is happening for the frames 52 and above because of the same reason
of approximately constant distance between the ego and the preceding vehicle.

Let's take a look at `HARRIS`-`ORB` pair alone:

<img src=results/FP6/Harris_ORB.png width=1000>

<img src=results/FP6/Harris_ORB_6_7.png width=1000>

As it can be seen it is missing a lot of data, including either nans or infs.
Nans are happening because no keypoint matches are found in the bounding box.

On `SHITOMASI`-`FREAK` pair lesser problem such as above can be seen:

<img src=results/FP6/SHI-FREAK.png width=1000>

<img src=results/FP6/SHI-FREAK_6_7.png width=1000>

I believe the reason is few matches are found inside the bounding box and most of 
them are on the far corners of the bounding box and they filtered out by the shrinking
bounding box.

The solution that it comes to my mind is that, we can do a lot better if we would
have ran the keypoint detection only on the bounding boxes instead of the whole image.
I believe a lot of algorithms would have given a better results in this case but of course
the algorithm would be slower.