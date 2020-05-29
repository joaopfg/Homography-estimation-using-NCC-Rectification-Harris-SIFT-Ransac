#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with H check
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(int argc, char* argv[])
{
    Mat img1 = imread("../IMG_0045.JPG");
    Mat img2 = imread("../IMG_0046.JPG");

    //Detect keypoints and compute descriptors using AKAZE
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    //Use brute-force matcher to find 2-nn matches
    //We use Hamming distance, because AKAZE uses binary descriptor by default
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    //Use 2-nn matches and ratio criterion to find correct keypoint matches
    vector<KeyPoint> matched1, matched2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    //-- Localize the object
    vector<Point2f> pts1;
    vector<Point2f> pts2;

    for(size_t i = 0; i < nn_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      DMatch first = nn_matches[i][0];
      pts1.push_back( kpts1[first.queryIdx].pt);
      pts2.push_back( kpts2[first.trainIdx].pt);
    }

    Mat H = findHomography(pts1, pts2, RANSAC);
    
    //Check if our matches fit in the H model
  
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = H * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    Mat img1Warped;
    warpPerspective(img1, img1Warped, H, img2.size());
    Mat halfLeft = img1Warped(Rect(0, 0, img1Warped.cols/2, img1Warped.rows));

    Mat img2Warped;
    warpPerspective(img2, img2Warped, H, img1.size());
    Mat halfRight = img2Warped(Rect(0, 0, img2Warped.cols/2, img2Warped.rows));

    Mat combined;
    hconcat(halfLeft, halfRight, combined);
    
    //Output results
    
    Mat res, res1, res2;

    drawKeypoints( img1, kpts1, res1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imwrite("../keypoints1.png", res1);
    drawKeypoints( img2, kpts2, res2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imwrite("../keypoints2.png", res2);

    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("../akaze_result.png", res);

    imshow("Images warped by homography", combined);
    imshow("keypoints1", res1);
    imshow("keypoints2", res2);
    imshow("matches", res);

    waitKey();  
    return 0;
} 

//Uncomment this main and comment the other if you want to make the tests with the rotated image

/*
int main(int argc, char* argv[])
{
    Mat img1 = imread("../IMG_0045.JPG");
    Mat img2 = imread("../IMG_0046r.JPG");

    //Detect keypoints and compute descriptors using AKAZE
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    //Use brute-force matcher to find 2-nn matches
    //We use Hamming distance, because AKAZE uses binary descriptor by default
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    //Use 2-nn matches and ratio criterion to find correct keypoint matches
    vector<KeyPoint> matched1, matched2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    //-- Localize the object
    vector<Point2f> pts1;
    vector<Point2f> pts2;

    for(size_t i = 0; i < nn_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      DMatch first = nn_matches[i][0];
      pts1.push_back( kpts1[first.queryIdx].pt);
      pts2.push_back( kpts2[first.trainIdx].pt);
    }

    Mat H = findHomography(pts1, pts2, RANSAC);
    
    //Check if our matches fit in the H model
  
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;
        col = H * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    
    //Output results
    
    Mat res, res1, res2;

    drawKeypoints( img1, kpts1, res1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imwrite("../keypoints1.png", res1);
    drawKeypoints( img2, kpts2, res2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imwrite("../keypoints2.png", res2);

    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("../akaze_result.png", res);

    imshow("keypoints1", res1);
    imshow("keypoints2", res2);
    imshow("matches", res);

    waitKey();  
    return 0;
}  */