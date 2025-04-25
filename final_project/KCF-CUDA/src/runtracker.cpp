#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>
#include <time.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
  cout << "ECE569 v1 normalizeAndTruncate Cuda optimization. 2 Kernel version" << endl;
  cout << "Operating at " << fixed << CLOCKS_PER_SEC << " ticks per second (important for time.h)." << endl;
  float tick_rate = CLOCKS_PER_SEC / 1000;

  if (argc > 5){
    cout << "argc > 5\n";
      
      return -1;
  }
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "config/image.txt";
  	listFile.open(fileName.c_str());

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
	string groundtruth = "config/region.txt";
  	groundtruthFile.open(groundtruth.c_str());

  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
	

  	istringstream ss(firstLine);

  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 
	cout << "Region of interest:" << endl
	     << "(x1, y1) = (" << setprecision(0) << x1 << ", " << y1  << ")" << endl
	     << "(x2, y2) = (" << x2 << ", " << y2 << ")" << endl
	     << "(x3, y3) = (" << x3 << ", " << y3 << ")" << endl
	     << "(x4, y4) = (" << x4 << ", " << y4 << ")" <<  endl;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	
	// Read Images
	ifstream listFramesFile;
	string listFrames = "config/image.txt";
	listFramesFile.open(listFrames.c_str());
	string frameName;


	// Write Results
	ofstream resultsFile;
	string resultsPath = "config/outputs/output.txt";
	resultsFile.open(resultsPath.c_str());
	resultsFile << "Hello World!\n";
	// Frame counter
	int nFrames = 0;
	float totalTime = 0;
	float frameTime = 0;

	fstream timingFile;
	string timingPath = "config/outputs/timingBuffer.txt";
	timingFile.open(timingPath.c_str(), ios::out);
	clock_t t = clock();
	
	//Match 1k video... not sure why HPC behaves strangely here... mess with window values. Keep 16x9 aspect ratio!
	if (!SILENT) {
	  namedWindow("Image", WINDOW_NORMAL);
	  resizeWindow("Image", 2560, 1440);
	}

	while ( getline(listFramesFile, frameName) ){
		
	        frameName = frameName;

		// Read each frame from the list
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		
		
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else {
			result = tracker.update(frame);
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		// ECE569: Adding timer!
		frameTime = (clock() - t) / tick_rate;
		timingFile << "Frame " << nFrames << " processed in " << setprecision(9) << frameTime  << " milliseconds.\n";
		t = clock();
		totalTime += frameTime;
		
		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(1);
		}
	}

	timingFile.close();

	// Add total time to file, then add frame-by-frame time
	ofstream totalTimingFile;
	string timingBuffer;
	totalTimingFile.open("config/outputs/timing.txt");
	timingFile.open(timingPath.c_str(), ios::in);
	totalTimingFile << "Total execution time: " << fixed << setprecision(1)  << totalTime << " ms.\n";
	
	while (getline(timingFile, timingBuffer)) {
	  totalTimingFile << timingBuffer << endl;
	}
	
	// Print total time
	cout  << "Total execution time: " << setprecision(1)  << totalTime << " ms.\n";
	
	// Cleaning up output mess
	timingFile.close();
	remove("config/outputs/timingBuffer.txt");
	resultsFile.close();
	listFile.close();
	totalTimingFile.close();
	}
