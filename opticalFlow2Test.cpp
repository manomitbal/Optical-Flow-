#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

static const double pi = 3.14159265358979323846;
#define MAX_COUNT 500

char rawWindow[] = "Raw Video";
char opticalFlowWindow[] = "Optical Flow Drift";
char trajectory[] = "Trajectory";
char fastfeat[] = "FAST Features";
char imageFileName[32];
long imageIndex = 0;
char keyPressed;


int main() 
{
	VideoCapture cap(1);
  double distSum = 0;
  double avgDist = 0;

 	Mat frame, grayFrames, rgbFrames, prevGrayFrame, originalFrame, copyrgbFrames;
  Mat H, H2;
         
  	
 
	
	Mat opticalFlow = Mat(cap.get(CV_CAP_PROP_FRAME_HEIGHT),
   	cap.get(CV_CAP_PROP_FRAME_HEIGHT), CV_32FC3);

	Mat traj = Mat(cap.get(CV_CAP_PROP_FRAME_HEIGHT), 
    cap.get(CV_CAP_PROP_FRAME_HEIGHT), CV_32FC3);

 	vector<Point2f> points1;
 	vector<Point2f> points2;
  vector<Point2f> mask;

 	Point preCenter = Point(500,500);

 	vector<uchar> status;
 	vector<float> err;
  vector<KeyPoint> keypoint;
  vector<DMatch> match;
  //vector<char> mask;

 	
	RNG rng(12345);
 	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
   	rng.uniform(0, 255));
 	bool init = true;

 	int i, k;
 	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
 	Size subPixWinSize(10, 10), winSize(31, 31);
 	namedWindow(rawWindow, CV_WINDOW_AUTOSIZE);
  namedWindow(opticalFlowWindow, CV_WINDOW_AUTOSIZE);
	//namedWindow(fastfeat, CV_WINDOW_AUTOSIZE);
       // namedWindow(trajectory, CV_WINDOW_AUTOSIZE);
 	double angle;

 	while (1) 
	{
  		cap >> frame;
  		frame.copyTo(rgbFrames);
	//	frame.copyTo(copyrgbFrames);
	//	frame.copyTo(originalFrame);
  		cvtColor(rgbFrames, grayFrames, CV_BGR2GRAY);

  		if (init) 
		  {
   			goodFeaturesToTrack(grayFrames, points1, MAX_COUNT, 0.01, 5, Mat(),
     			3, 1, 0.04);
		//	FAST(grayFrames, keypoint, 30, true);
   			init = false;
  		} 
		
		  else if (!points2.empty()) 
		  {
   			cout << "\n\n\nCalculating  calcOpticalFlowPyrLK\n\n\n\n\n";
   			calcOpticalFlowPyrLK(prevGrayFrame, grayFrames, points2, points1,
     			status, err, winSize, 3, termcrit, 0, 0.001);
        H = findFundamentalMat(points1, points2, CV_FM_RANSAC, 3, 0.99);
			 
   			for (i = k = 0; i < points2.size(); i++) 
			  {
				      int dx = int(points1[i].x - points2[i].x);
              int dy = int(points1[i].y - points2[i].y);
              
              cout << "Detected Feature Drift (Threshold to Hover)  X :  "
      				<< int(points1[i].x - points2[i].x) << "\t Y :  "
      				<< int(points1[i].y - points2[i].y) << "\t\t" << i
      				<< "\n" ;
              
              Point center = Point (dx,dy);
              //line(traj, preCenter, center, Scalar(255,0,0), 1, CV_AA, 0);
              preCenter = Point(center);
    				
				
				if ((points1[i].x - points2[i].x) > 0) 
				{
     					
					
					double angle, hypotenuse;
					angle = atan2((double) points1[i].y - points2[i].y, (double) points1[i].x - points2[i].x);
					
					circle(rgbFrames, points1[i], 2, Scalar(0, 0, 255), 2, 1,0);

					//circle(traj, points1[i], 2,Scalar(0,0,255), 2,1,0);
    					

			//	FAST(grayFrames, keypoints, 40, true);
					arrowedLine(rgbFrames, points1[i], points2[i], Scalar(0, 0, 255),
       					2,CV_AA, 0, 0.2);
					
					//drawKeypoints(grayFrames, keypoint, copyrgbFrames, Scalar::all(-1), 4);
					
					
					
					line(opticalFlow, points1[i], points2[i], Scalar(0, 0, 255),
       					1, CV_AA, 0);
     					circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1,
       					0);
    				} 
				
				else
				{
     					

			//		FAST(grayFrames, keypoints, 1, true);
					arrowedLine(rgbFrames, points1[i], points2[i], Scalar(0, 255, 0),
       					2, CV_AA, 0, 0.2);
					//drawKeypoints(grayFrames, keypoint, copyrgbFrames, Scalar::all(-1), 4);
     					circle(rgbFrames, points1[i], 2, Scalar(0, 0, 255), 2, 1,
       					0);
              //circle(traj, points1[i],2, Scalar(0,0,255), 2,1,0);

     					line(opticalFlow, points1[i], points2[i], Scalar(0, 255, 0),
       					1, CV_AA, 0);
     					circle(opticalFlow, points1[i], 1, Scalar(255, 0, 0), 1, 1,
       					0);
    				}	
				
    				points1[k++] = points1[i];

   			}

   			goodFeaturesToTrack(grayFrames, points1, MAX_COUNT, 0.01, 10, Mat(),
     			3, 1, 0.04);
        //H2 = findHomography(points1, points2, 0, 3);
        //H = findFundamentalMat(points1, points2, CV_FM_RANSAC, 3, 0.99);
			
                  //      FAST(grayFrames, keypoint, 30, true);


  		}

  		imshow(rawWindow, rgbFrames);
  		imshow(opticalFlowWindow, opticalFlow);
    //imshow(trajectory, H);       
		//imshow(fastfeat, copyrgbFrames);         

  		swap(points2, points1);
  		points1.clear();
  		grayFrames.copyTo(prevGrayFrame);

  		keyPressed = waitKey(10);
  		if (keyPressed == 27) 
		  {
   			break;
  		} 
		  else if (keyPressed == 'r') 
		  {
   			opticalFlow = Mat(cap.get(CV_CAP_PROP_FRAME_HEIGHT),
     		cap.get(CV_CAP_PROP_FRAME_HEIGHT), CV_32FC3);
  		}

 	}
}

