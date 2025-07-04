/*
 * @Author: Strinkin 2638578339@qq.com
 * @Date: 2025-04-05 22:51:04
 * @LastEditors: Strinkin 2638578339@qq.com
 * @LastEditTime: 2025-04-09 22:54:29
 * @FilePath: /undefined/home/strinkin/cpp_workspace/imu_hik_viz/camera/include/HikGeneral.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef __HIKGENERAL_HPP__
#define __HIKGENERAL_HPP__
/**
 * @author: strinkin
 * @date: 2025-1-6
 * @details 不使用cuda
*/
#include "interface/HikVision.hpp"


namespace camera
{

class HikGeneral : public HikVision {
    public:
        using HikVision::initCamera;
        using HikVision::releaseCamera;
        using HikVision::setCameraParams;
        void startGrabing() override;
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
void HikGeneral::startGrabing() {

    // 开始取流
    HikVision::nRet = MV_CC_StartGrabbing(HikVision::handle);

    // 获取数据包大小
    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    HikVision::nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    HikVision::nDataSize = stParam.nCurValue;

    
}

void HikGeneral::bindImage(unsigned char** img_ptr, int step) {
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
 * @details 通用版本不需要使用前两个参数
*/
void HikGeneral::getOneFrame(unsigned char* img_ptr, int step, int time_out) {
    HikVision::nRet = MV_CC_GetOneFrameTimeout(HikVision::handle, HikVision::pData, nDataSize, &stImageInfo, time_out);
    if (HikVision::nRet == MV_OK) {
        // printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n", 
        // stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
    } else {
        printf("No data[%x]\n", HikVision::nRet);
    }   
}
} // namespace camera

#endif // __HIKGENERAL_HPP__