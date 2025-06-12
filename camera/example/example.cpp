#include "HikCuda.hpp"
#include "HikGeneral.hpp"
#include "HikCudaV2.hpp"
#include "HikGeneralV2.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thread>

using namespace std;

void cudaDemo();
void generalDemo();
void cudaDemoV2();
void generalDemoV2();

int main() {
    cudaDemo();
    generalDemo();
    cudaDemoV2();
    generalDemoV2();
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
        auto start = std::chrono::high_resolution_clock::now();
        cam.getOneFrame((uchar*)d_img.cudaPtr(), d_img.step, 5);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "getOneFrame time: " << duration.count() << "us" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    cudaFreeHost(p);
    cudaFree(d_img.cudaPtr());
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
        auto start = std::chrono::high_resolution_clock::now();
        cam.getOneFrame(nullptr, 0, 5);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "getOneFrame time: " << duration.count() << "us" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
    free(img1.data);
    cam.releaseCamera();
}

void cudaDemoV2() {
    camera::HikCudaV2 cam;
    cam.initCamera();
    cam.setCameraParams();
    cam.startGrabing();
    cam.allocMemForpData();


    MV_FRAME_OUT_INFO_EX* info = cam.getImageInfo();
    cv::cuda::GpuMat d_img, d_img3;
    unsigned char* p = cam.getpData();
    d_img.create(info->nHeight, info->nWidth, CV_8UC1);
    cudaMalloc(&d_img.data, sizeof(unsigned char) * info->nHeight * d_img.step);
    cv::Mat img3;
    
    int frameCount = 0;
    double fps = 0.0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cam.getOneFrame(5);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "getOneFrame time: " << duration.count() << "us" << std::endl;
        cudaMemcpy2D(d_img.data, d_img.step, p, info->nWidth, info->nWidth, info->nHeight, cudaMemcpyHostToDevice);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

        cv::putText(img3, "cudaDemoV2: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("img3", img3);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cam.releaseCamera(); 
    cam.releaseMemForpData();
    cudaFree(d_img.cudaPtr());
}

void generalDemoV2() {
    camera::HikGeneralV2 cam;
    cam.initCamera();
    cam.setCameraParams();
    cam.startGrabing();
    cam.allocMemForpData();

    MV_FRAME_OUT_INFO_EX* info = cam.getImageInfo();
    unsigned char* p = cam.getpData();
    cv::Mat img1(info->nHeight, info->nWidth, CV_8UC1, p);
    cv::Mat img3;

    int frameCount = 0;
    double fps = 0.0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (1) {
        auto start = std::chrono::high_resolution_clock::now();
        cam.getOneFrame(5);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "getOneFrame time: " << duration.count() << "us" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

        cv::putText(img3, "generalDemoV2: " + to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("img3", img3);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cam.releaseMemForpData();
    cam.releaseCamera();
}
