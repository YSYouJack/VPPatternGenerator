//
//  main.cpp
//  OpenCVTest
//
//  Created by Ying-ShiuanYou on 2021/4/5.
//

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

int main(int argc, const char * argv[])
{
    if (3 != argc) {
        std::cout << "Usage: OpenCVTest in_dir out_dir" << std::endl;
        return EXIT_FAILURE;
    }
    
    fs::path inDir = argv[1];
    if (!fs::is_directory(inDir)) {
        std::cerr << "Error: Cannot find input directory!" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<fs::path> inputImagePathes;
    for (auto & p: fs::directory_iterator(inDir)) {
        if (!fs::is_regular_file(p)) {
            continue;
        }
        
        auto ext = p.path().extension().string();
        for (size_t i = 0; i < ext.size(); ++i) {
            ext[i] = std::tolower(ext[i]);
        }
        
        if (".jpeg" == ext ||
            ".jpg" == ext ||
            ".png" == ext ||
            ".tif" == ext ||
            ".tiff" == ext)
        {
            inputImagePathes.emplace_back(p);
        }
    }
    
    if (inputImagePathes.size() < 2) {
        std::cerr << "Error: The valid images is not enough!" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<std::pair<size_t, size_t>> imagePairs;
    for (size_t i = 0; i < inputImagePathes.size(); ++i) {
        for (size_t j = i + 1; j < inputImagePathes.size(); ++j) {
            imagePairs.emplace_back(std::make_pair(i, j));
        }
    }
    
    fs::path outDir = argv[2];
    if (!fs::exists(outDir)) {
        fs::create_directory(outDir);
    }
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(imagePairs.begin(), imagePairs.end(), std::default_random_engine(seed));
    
    // A4 Size: https://www.info.print-print.co.uk/guide-to-image-resolution/
    cv::Mat bigImg(3508, 2480, CV_8UC1);
    
    size_t outCnt = (std::max)(static_cast<size_t>(1), imagePairs.size() / 2);
    for (size_t i = 0; i < outCnt; ++i) {
        const auto &path0 =inputImagePathes[imagePairs[i].first];
        const auto &path1 =inputImagePathes[imagePairs[i].second];
        
        bigImg.setTo(cv::Scalar(255, 255, 255, 255));
        
        cv::Mat img0 = cv::imread(path0.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat img1 = cv::imread(path1.string(), cv::IMREAD_GRAYSCALE);
        
        cv::threshold(img0, img0, 32, 255, cv::THRESH_BINARY);
        cv::threshold(img1, img1, 32, 255, cv::THRESH_BINARY);
        
        cv::Rect roi0 = cv::boundingRect(img0);
        cv::Rect roi1 = cv::boundingRect(img1);
        
        const int BOUNDARY_SIZE = 250;
        const int GRID_COLUMNS = 5;
        const int GRID_ROWS = 5;
        
        int gridW = (bigImg.cols - 2 * BOUNDARY_SIZE) / GRID_COLUMNS;
        int gridH = (bigImg.rows - 2 * BOUNDARY_SIZE) / (GRID_ROWS + 2);
        int gridLen = std::min(gridW, gridH);
        
        cv::Mat smallImg0;
        cv::Mat smallImg1;
        cv::resize(img0, smallImg0, cv::Size(gridLen - 2, gridLen - 2));
        cv::resize(img1, smallImg1, cv::Size(gridLen - 2, gridLen - 2));
        
        std::vector<cv::Mat> cellImgs(2);
        for (auto & m : cellImgs) {
            m = cv::Mat(gridLen, gridLen, smallImg0.type());
        }
        
        cv::copyMakeBorder(smallImg0, cellImgs[0], 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::copyMakeBorder(smallImg1, cellImgs[1], 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        
        int offsetX = (bigImg.cols - gridLen * GRID_COLUMNS) / 2;
        int offsetY = (bigImg.rows - gridLen * (GRID_ROWS + 2)) / 2;
        
        std::vector<int> ids(GRID_ROWS * GRID_COLUMNS);
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<int>(i % 2);
        }
        
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
        
        for (int i = 0; i < GRID_ROWS; ++i) {
            int top = (i + 2) * gridLen + offsetY;
            for (int j = 0; j < GRID_COLUMNS; ++j) {
                int left = j * gridLen + offsetX;
                cv::Rect r(left, top, gridLen, gridLen);
                cellImgs[ids[i * GRID_COLUMNS + j]].copyTo(bigImg(r));
            }
        }
        
        cellImgs[ids[ids.size() / 2]].copyTo(bigImg(cv::Rect(offsetX, offsetY, gridLen, gridLen)));
        
        std::stringstream ss;
        ss << i << ".jpg";
        
        fs::path outPath = outDir / ss.str();
        
        ss.str("");
        
        cv::imwrite(outPath.string(), bigImg);
    }
    
    return 0;
}
