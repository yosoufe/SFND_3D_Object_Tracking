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

<img align="middle" src="https://latex.codecogs.com/gif.latex?TTC&space;=&space;\frac{-\Delta&space;t}{1-\frac{h_1}{h_0}}" title="TTC = \frac{-\Delta t}{1-\frac{h_1}{h_0}}" />

<a name="FP3" />

## FP3 - Associate Keypoint Correspondences with Bounding Boxes

<a name="FP4" />

## FP4 - Compute Camera-based TTC

<a name="FP5" />

## FP5 - Performance Evaluation 1

<a name="FP6" />

## FP6 - Performance Evaluation 2
