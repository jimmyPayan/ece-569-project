#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h> // for mkdir
#include <sys/types.h>
#include <errno.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./video_to_frames <video_path>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    std::string outputDir = "img";
    std::string listFile = "images.txt";

    // Create output directory if it doesn't exist
    struct stat st;
    if (stat(outputDir.c_str(), &st) != 0) {
        if (mkdir(outputDir.c_str(), 0755) != 0) {
            std::cerr << "Error: Could not create directory " << outputDir << std::endl;
            return -1;
        }
    }

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << videoPath << std::endl;
        return -1;
    }

    int frameCount = 0;
    cv::Mat frame;
    std::ofstream listOut(listFile.c_str());

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        std::ostringstream filename;
        filename << outputDir << "/"
                 << std::setw(4) << std::setfill('0') << frameCount << ".jpg";

        if (!cv::imwrite(filename.str(), frame)) {
            std::cerr << "Warning: Failed to save frame " << frameCount << std::endl;
            continue;
        }

        listOut << filename.str() << std::endl;
        frameCount++;
    }

    std::cout << "Extracted " << frameCount << " frames." << std::endl;

    cap.release();
    listOut.close();

    return 0;
}
