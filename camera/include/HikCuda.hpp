#pragma once
/**
 * @author: strinkin
 * @date: 2025-1-6
 * @details: 使用cuda
*/
#include "interface/HikVision.hpp"
#include <cuda_runtime.h>

namespace camera
{

class HikCuda : public HikVision {
    public:
        using HikVision::initCamera;
        using HikVision::releaseCamera;
        using HikVision::setCameraParams;
        void startGrabing() override;
        void bindImage(unsigned char** img_ptr, int step) override;
        void getOneFrame(
            unsigned char* img_ptr,
            int step, 
            int time_out = 100) override;
};

void HikCuda::startGrabing() {
    /**
     * @override
     * @param data_host: cv::Mat->ptr()
    */
    // 开始取流
    HikVision::nRet = MV_CC_StartGrabbing(HikVision::handle);

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    HikVision::nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    HikVision::nDataSize = stParam.nCurValue;
}

void HikCuda::bindImage(unsigned char** img_ptr, int step) {
    /**
     * @details CUDA版本 
     * @param img_ptr: cv::GpuMat->ptr()
    */
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

void HikCuda::getOneFrame(unsigned char* img_ptr, int step, int time_out) {
    /**
     * @details CUDA版本
     * @param img_ptr: cv::GpuMat->ptr()
     * @param img_step: cv::GpuMat->step() | num of bytes per row = width * channels
     * @param time_out: 抓图超时时间
     * @example 1. GpuMat d_img(src.rows, src.cols, CV_8UC1);
     * @example 2. cudaMemcpy2D(d_img.cudaPtr(), d_img.step, pData, 640, 640, 480, cudaMemcpyHostToDevice);
    */
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

