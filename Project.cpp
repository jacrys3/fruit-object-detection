#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <string.h>
#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;


// training values
// Apple cascade: -w 24 -h 24 -numPos 100 -numNeg 1000 -numStages 8 -maxFalseAlarmRate .3 -minHitRate 0.999
// Banana cascade : -w 24 -h 24 -numPos 100 -numNeg 300 -numStages 15 -maxFalseAlarmRate .4 -minHitRate 0.999
// Orange cascade : -w 24 -h 24 -numPos 120 -numNeg 500 -numStages 12 -maxFalseAlarmRate .35 -minHitRate 0.999



// generateNegativeDescriptionFile 
// used to create the negative description files for ease of use to the user
void generateNegativeDescriptionFile(fs::path& path)
{
	// create and open a text file
	ofstream myFile("negative.txt");

	if (!myFile.is_open())
	{
		cout << "error" << endl;
		return;
	}

	cout << "in" << endl;

	// iterate through each file in the directory 
	for (auto const& dir_entry : fs::directory_iterator{ path })
	{
		string filename = dir_entry.path().filename().string();
		myFile << "negative/" << filename << "\n";
	}
}


int main(int argc, char *argv[])
{
	/** take in input from the batch file
	*	variables include:
	*		-numImages=#   specify the amount of images to be created (max 25)
	**/

	int numImages = -1; // num of images up to 25


	string num = "";
	if (argc > 1)
	{
		num += argv[1][11];
		if (argv[1][12] != NULL) num += argv[1][12];
		numImages = stoi(num);
	}
	
	//fs::path p("C:\\UWB_Year_3\\CSS_487\\FinalProject\\FinalProject\\negative");
	//generateNegativeDescriptionFile(p);

	for (int i = 1; i <= numImages; i++)
	{	
		cout << "running object detection on image " << i << endl;
		Mat testImage = imread("test/mixed_" + to_string(i) + ".jpg");

		Mat image_gray;
		cvtColor(testImage, image_gray, COLOR_BGR2GRAY);
		//imwrite("gray.jpg", image_gray);

		Mat image_rgb;
		cvtColor(testImage, image_rgb, COLOR_BGR2RGB);


		CascadeClassifier apple_cascade = CascadeClassifier("cascade_apple/cascade.xml");
		CascadeClassifier banana_cascade = CascadeClassifier("cascade_banana/cascade.xml");
		CascadeClassifier orange_cascade = CascadeClassifier("cascade_orange/cascade.xml");


		vector<Rect> apples;
		apple_cascade.detectMultiScale(image_gray, apples, 1.1, 3, 0, Size(3, 3));

		vector<Rect> bananas;
		banana_cascade.detectMultiScale(image_gray, bananas, 1.1, 3, 0, Size(3, 3));

		vector<Rect> oranges;
		orange_cascade.detectMultiScale(image_gray, oranges, 1.1, 3, 0, Size(3, 3));


		if (apples.size() == 0) cout << "cannot find any apples" << endl;
		if (bananas.size() == 0) cout << "cannot find any bananas" << endl;
		if (oranges.size() == 0) cout << "cannot find any oranges" << endl;
	

		for (Rect apple : apples)
		{
			putText(image_rgb, "Apple", Point(apple.x + 5, apple.y - 15), FONT_HERSHEY_DUPLEX, .5, Scalar(0, 255, 0), 2);
			rectangle(image_rgb, Point(apple.x, apple.y), Point(apple.x + apple.height, apple.y + apple.width), Scalar(0, 255, 0), 3);
		}

		for (Rect banana : bananas)
		{
			putText(image_rgb, "Banana", Point(banana.x + 5, banana.y - 15), FONT_HERSHEY_DUPLEX, .5, Scalar(150, 50, 100), 2);
			rectangle(image_rgb, Point(banana.x, banana.y), Point(banana.x + banana.height, banana.y + banana.width), Scalar(150, 50, 100), 3);
		}

		for (Rect orange : oranges)
		{
			putText(image_rgb, "Orange", Point(orange.x + 5, orange.y - 15), FONT_HERSHEY_DUPLEX, .5, Scalar(0, 0, 255), 2);
			rectangle(image_rgb, Point(orange.x, orange.y), Point(orange.x + orange.height, orange.y + orange.width), Scalar(0, 0, 255), 3);
		}

		Mat output;
		cvtColor(image_rgb, output, COLOR_RGB2BGR);
		imwrite("output" + to_string(i) + ".jpg", output);

		imshow("output" + to_string(i), output);
		waitKey(0);

		destroyAllWindows();
	}

	return 0;
}