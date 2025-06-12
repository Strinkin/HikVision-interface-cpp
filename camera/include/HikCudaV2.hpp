#ifndef __HIKCUDAV2_HPP__
#define __HIKCUDAV2_HPP__
/**
 * @author: strinkin
 * @date: 2025-1-6
 * @details: 使用cuda
*/
#include "interface/HikVision.hpp"
#include <cuda_runtime.h>

namespace camera
{

class HikCudaV2 : public HikVision {
    public:
        ~HikCudaV2() {
            releaseMemForpData();
        }
        using HikVision::initCamera;
        using HikVision::releaseCamera;
        using HikVision::setCameraParams;
        void startGrabing() override;
        void allocMemForpData();
        void releaseMemForpData();
        unsigned char* getpData();
        bool getOneFrame(int time_out); 
        void bindImage(unsigned char** img_ptr, int step) override;
        void getOneFrame(
            unsigned char* img_ptr,
            int step, 
            int time_out = 100);
};

/**
 * @override
 * @param data_host: cv::Mat->ptr()
*/
void HikCudaV2::startGrabing() {
    // 开始取流
    HikVision::nRet = MV_CC_StartGrabbing(HikVision::handle);

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    HikVision::nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    HikVision::nDataSize = stParam.nCurValue;
}

void HikCudaV2::allocMemForpData() {
    cudaMallocHost(&pData, sizeof(unsigned char) * HikVision::nDataSize * HikVision::channels);
    int tolerance = 20;
    do 
    {
        // 获取图像信息
        HikVision::stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, HikVision::nDataSize, &stImageInfo, 100);
        if (HikVision::nRet == MV_OK) {
            break;
        }
    } while(tolerance--);
}

void HikCudaV2::releaseMemForpData() {
    cudaFreeHost(HikVision::pData);
}

unsigned char* HikCudaV2::getpData() {
    return HikVision::pData;
}

bool HikCudaV2::getOneFrame(int time_out = 10) {
    HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, time_out);
    if (HikVision::nRet == MV_OK) {
        // printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n", 
        // stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
        return true;
    } else {
        printf("No data[%x]\n", HikVision::nRet);
        return false;
    }   
}


// ##########################################

/**
 * 弃用
 * @details CUDA版本 
 * @param img_ptr: cv::GpuMat->ptr()
*/
void HikCudaV2::bindImage(unsigned char** img_ptr, int step) {

    cudaMallocHost(&pData, sizeof(unsigned char) * nDataSize * channels); // Host pinned Memory
    cudaMalloc(img_ptr, sizeof(unsigned char) * step * HikVision::stImageInfo.nHeight * channels); // Device Memory
    do 
    {
        // 获取图像信息
        HikVision::stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, HikVision::nDataSize, &stImageInfo, 1000);
        if (HikVision::nRet == MV_OK) {
            break;
        }
    } while(1);
}

/**
 * 弃用
 * @details CUDA版本
 * @param img_ptr: cv::GpuMat->ptr()
 * @param img_step: cv::GpuMat->step() | num of bytes per row = width * channels
 * @param time_out: 抓图超时时间
 * @example 1. GpuMat d_img(src.rows, src.cols, CV_8UC1);
 * @example 2. cudaMemcpy2D(d_img.cudaPtr(), d_img.step, pData, 640, 640, 480, cudaMemcpyHostToDevice);
*/
void HikCudaV2::getOneFrame(unsigned char* img_ptr, int step, int time_out) {
    HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, time_out);
    if (HikVision::nRet == MV_OK) {
        cudaMemcpy2D(img_ptr, step, pData, stImageInfo.nWidth * channels, stImageInfo.nWidth, stImageInfo.nHeight, cudaMemcpyHostToDevice);
        // printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n", 
        // stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
    } else {
        printf("No data[%x]\n", HikVision::nRet);
    }   
}
} // namespace camera

#endif // __HIKCUDAV2_HPP__