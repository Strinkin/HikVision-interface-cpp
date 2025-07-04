#ifndef __HIKGENRALV2_HPP__
#define __HIKGENRALV2_HPP__
/**
 * @author: strinkin
 * @date: 2025-1-6
 * @details 不使用cuda
*/
#include "interface/HikVision.hpp"


namespace camera
{

class HikGeneralV2 : public HikVision {
    public:
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
            unsigned char* img_ptr = nullptr, 
            int step = 0, 
            int time_out = 100);
};

/**
 * @override
 * @param data_host: cv::Mat->ptr()
*/
void HikGeneralV2::startGrabing() {

    // 开始取流
    HikVision::nRet = MV_CC_StartGrabbing(HikVision::handle);

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    HikVision::nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    HikVision::nDataSize = stParam.nCurValue;

}

void HikGeneralV2::allocMemForpData() {
    HikVision::pData = (unsigned char*)malloc(HikVision::nDataSize * HikVision::channels);
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

void HikGeneralV2::releaseMemForpData() {
    free(HikVision::pData);
}

unsigned char* HikGeneralV2::getpData() {
    return HikVision::pData;
}

bool HikGeneralV2::getOneFrame(int time_out = 5) {
    HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, time_out);
    if (HikVision::nRet == MV_OK) {
        return true;
    } else {
        printf("No data[%x]\n", HikVision::nRet);
        return false;
    }
}

// #############################################

// 弃用
void HikGeneralV2::bindImage(unsigned char** img_ptr, int step) {
    HikVision::pData = (unsigned char *)malloc(sizeof(unsigned char) * HikVision::nDataSize);
    do 
    {   
        // 获取图像信息
        HikVision::stImageInfo = {0};
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, 1000);
        if (HikVision::nRet == MV_OK) {
            *img_ptr = HikVision::pData;
            break;
        }
    } while(1);
}

/**
 * 弃用
 * @details 通用版本不需要使用前两个参数
*/
void HikGeneralV2::getOneFrame(unsigned char* img_ptr, int step, int time_out) {
    HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, time_out);
    if (HikVision::nRet == MV_OK) {
        // printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n", 
        // stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
    } else {
        printf("No data[%x]\n", HikVision::nRet);
    }   
}
} // namespace camera

#endif // __HIKGENRALV2_HPP__