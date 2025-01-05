#include "HikCuda.hpp"
#include "HikGeneral.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;

void cudaDemo();
void generalDemo();

int main() {
    cudaDemo();
    generalDemo();
    return 0;
}

void cudaDemo() {
    camera::HikCuda cam;
    cam.initCamera();
    cam.setCameraParams();
    cam.startGrabing();

    MV_FRAME_OUT_INFO_EX* info = cam.getImageInfo();
    cv::cuda::GpuMat d_img, d_img3;
    uint8_t* p;
    d_img.create(info->nHeight, info->nWidth, CV_8UC1);

    cam.bindImage(&p, d_img.step);
    d_img.data = p;
    cv::Mat img3;
    
    int frameCount = 0;
    double fps = 0.0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1)
    {
        cam.getOneFrame((uchar*)d_img.cudaPtr(), d_img.step, 5);
        cv::cuda::demosaicing(d_img, d_img3, cv::COLOR_BayerRG2BGR);
        d_img3.download(img3);


        // Increment frame count
        frameCount++;
        // Calculate elapsed time and FPS every second
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - startTime;
        if (elapsedTime.count() >= 1.0) {
            fps = frameCount / elapsedTime.count(); // Calculate FPS
            // std::cout << "FPS: " << fps << std::endl; // Print FPS
            
            // Reset for next second
            frameCount = 0;
            startTime = currentTime;
        }

        cv::putText(img3, "cudaDemo: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("img3", img3);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cam.releaseCamera();
    cudaFree(p);
}

void generalDemo() {
    camera::HikGeneral cam;
    cam.initCamera();
    cam.setCameraParams();
    cam.startGrabing();

    MV_FRAME_OUT_INFO_EX* info = cam.getImageInfo();
    cv::Mat img1(info->nHeight, info->nWidth, CV_8UC1);
    cv::Mat img3;
    cam.bindImage(&img1.data, 0);

    int frameCount = 0;
    double fps = 0.0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1) {
        cam.getOneFrame(nullptr, 0, 5);
        cv::demosaicing(img1, img3, cv::COLOR_BayerBG2BGR);

         // Increment frame count
        frameCount++;
        // Calculate elapsed time and FPS every second
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - startTime;
        if (elapsedTime.count() >= 1.0) {
            fps = frameCount / elapsedTime.count(); // Calculate FPS
            // std::cout << "FPS: " << fps << std::endl; // Print FPS
            
            // Reset for next second
            frameCount = 0;
            startTime = currentTime;
        }

        cv::putText(img3, "generalDemo: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("img3", img3);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}