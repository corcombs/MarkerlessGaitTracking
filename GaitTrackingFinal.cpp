#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "opencv/cv.h"
#include "math.h"
#include <queue>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{

    queue<double> angleChange;
    Mat back, frame, frame_diff;
    double angle,avgAngle=0,lastAngle=0;
    double angleChangeSum=0;
    VideoCapture capture;
    
    Point neck;
    Point torso;
    Point footA;
    Point footB;
    Point jointA;
    Point jointB;
    int neck_y,torso_y,joint_y,foot_y;
    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    BackgroundSubtractorMOG bg;
    vector<vector<Point> > contours;
    bool objectDetected=false;


    while(1){
        capture >> frame;

        
        bg.operator ()(frame,frame_diff, 0.1);
        bg.getBackgroundImage(back);
        threshold(frame_diff, frame_diff, 15, 255, cv::THRESH_BINARY);
        imshow("GMM FG (KadewTraKuPong&Bowden)", frame_diff);
        
        
        
        Mat dist;
        distanceTransform(frame_diff, dist, CV_DIST_L2, 3);
        normalize(dist, dist, 0, 1., NORM_MINMAX);
        
        findContours(frame_diff,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        
        if(contours.size()>10){
            objectDetected=true;
        }else{
            objectDetected = false;
        }
        
        if(objectDetected){
            vector<Point> contour;
            vector<Rect> boundRect( contours.size() );
            int area;
            Rect maxRect = boundingRect(Mat(contours[0]));
            Moments mu;
            Point2f center_of_gravity, aleg1,aleg2;
            int boxHeight = maxRect.height;
            int boxWidth = maxRect.width;
            
            // Get bounding rectangle
            for (int i=0; i < contours.size(); i++) {
                boundRect[i] = boundingRect( Mat(contours[i]));
                area = boundRect[i].height * boundRect[i].width;
                
                // There could be multiple smaller contours in the image
                // that's random noise. Only pick the largest one.
                if (area > (boxHeight * boxWidth)) {
                    maxRect = boundRect[i];
                    contour = contours[i];
                    
                    // calculate center of gravity
                    mu = moments(contours[i], false);
                    center_of_gravity = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );
                    boxHeight = maxRect.height;
                    boxWidth = maxRect.width;
                }
            }
            if(maxRect.height>=10 && maxRect.width>=10){
                int box_left = maxRect.tl().x;
                int box_top = maxRect.tl().y;
                int box_right = maxRect.br().x;
                int box_bottom = maxRect.br().y;
                int box_half = box_left + (box_right-box_left)/2;
                
                Mat temp;
                dist.copyTo(temp);
                
                neck_y = box_top + boxHeight * 0.13;
                torso_y = box_top + boxHeight * 0.53;
                joint_y = box_top + boxHeight*0.75;
                foot_y = box_bottom - boxHeight*0.05;
                minMaxLoc(temp.colRange(box_left, box_right).row(neck_y),NULL, NULL, NULL, &neck);
                minMaxLoc(temp.colRange(box_left, box_right).row(torso_y),NULL, NULL, NULL, &torso);
                minMaxLoc(temp.colRange(box_left, box_half).row(joint_y),NULL, NULL, NULL, &jointA);
                minMaxLoc(temp.colRange(box_half, box_right).row(joint_y),NULL, NULL, NULL, &jointB);
                minMaxLoc(temp.colRange(box_left, box_half).row(foot_y),NULL, NULL, NULL, &footA);
                minMaxLoc(temp.colRange(box_half, box_right).row(foot_y),NULL, NULL, NULL, &footB);
                
                neck.x += box_left;
                neck.y = neck_y;
                torso.x += box_left;
                torso.y = torso_y;
                jointA.x += box_left;
                jointA.y = joint_y;
                jointB.x += box_half;
                jointB.y = joint_y;
                footA.x += box_left;
                footA.y = foot_y;
                footB.x += box_half;
                footB.y = foot_y;
                //find leg angle
                aleg1=torso-footA;
                aleg2=torso-footB;
                angle=acos(((aleg1.x*aleg2.x)+(aleg1.y*aleg2.y))/(norm(aleg1)*norm(aleg2)))*180/M_PI;
                
                //add angles to queue and pop if the queue is bigger than 3
                if(angleChange.size()>3){
                    angleChange.push(abs(lastAngle-angle));
                    angleChangeSum=angleChangeSum+angleChange.back();
                    angleChangeSum=angleChangeSum-angleChange.front();
                    angleChange.pop();
                    avgAngle=angleChangeSum/angleChange.size();
                    lastAngle=angle;
                }else{
                    if(lastAngle!=0){
                        angleChange.push(abs(lastAngle-angle));
                        angleChangeSum=angleChangeSum+angleChange.back();
                        avgAngle=angleChangeSum/angleChange.size();
                    }
                    lastAngle=angle;
                }
                putText(frame, to_string(avgAngle), cvPoint(30,45),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
                if(avgAngle<18){
                    putText(frame, "Walking", cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
                }else{
                    putText(frame, "Running", cvPoint(30,30),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
                }
                
                rectangle( frame, maxRect, Scalar(0,255,0), 3);
                circle(frame, neck, 4, Scalar(255, 0, 255), 4);
                circle(frame, torso, 4, Scalar(255, 0, 255), 4);
                circle(frame, footA, 4, Scalar(255, 0, 255), 4);
                circle(frame, footB, 4, Scalar(255, 0, 255), 4);
                circle(frame, jointA, 4, Scalar(255, 0, 255), 4);
                circle(frame, jointB, 4, Scalar(255, 0, 255), 4);
                line(frame, torso, jointA, Scalar(255,0,255),1);
                line(frame, torso, jointB, Scalar(255,0,255),1);
                line(frame, jointA, footA, Scalar(255,0,255),1);
                line(frame, jointB, footB, Scalar(255,0,255),1);
                line(frame, neck, torso, Scalar(255,0,255),1);
            
            }
        }
        
        if(waitKey(30) >= 0) break;
        imshow("Distance Transform", dist);
        imshow("Frame", frame);
    }
    return 0;
}