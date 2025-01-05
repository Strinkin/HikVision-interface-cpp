#pragma once
/**
 * @author: strinkin
 * @date: 2025-1-6
 * @details 相机接口类
*/

#include "MvCameraControl.h"
#include "utils/getParam.hpp"

namespace fs = std::filesystem;

namespace camera {
    
class HikVision {
    protected:
        void* handle = nullptr; // 相机句柄
        int nRet = MV_OK; // SDK返回值
        unsigned char* pData; // 图像数据地址
        MV_FRAME_OUT_INFO_EX stImageInfo; // 图像信息
        unsigned int channels = 1; // 图像通道数
        unsigned int nDataSize; // 图像数据总字节数
    protected:
        bool initCamera();
        bool releaseCamera();
        void setCameraParams();

        virtual void startGrabing() = 0;
        virtual void bindImage(unsigned char** img_ptr, int step) = 0;
        virtual void getOneFrame(
            unsigned char* img_ptr, 
            int step, 
            int time_out ) = 0;

        void stopGrabing() { nRet = MV_CC_StopGrabbing(handle); }
    public:
        MV_FRAME_OUT_INFO_EX* getImageInfo() { return &stImageInfo; };
};

bool HikVision::initCamera() {
    // 初始化SDK
    HikVision::nRet = MV_CC_Initialize(); 
    if (MV_OK != HikVision::nRet) {
        printf("Initialize SDK fail! nRet [0x%x]\n", HikVision::nRet);
    }
    
    // 枚举设备
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    HikVision::nRet = MV_CC_EnumDevices(
        MV_GIGE_DEVICE | 
        MV_USB_DEVICE | 
        MV_GENTL_CAMERALINK_DEVICE | 
        MV_GENTL_CXP_DEVICE | 
        MV_GENTL_XOF_DEVICE, 
        &stDeviceList
    );
    if (stDeviceList.nDeviceNum > 0) {
        printf("[device %d]:\n", stDeviceList.nDeviceNum-1);     
    } else {
        printf("Find No Devices!\n");
        exit(-1);
    }
    
    // 选择设备并创建句柄
    unsigned int nIndex = 0;
    HikVision::nRet = MV_CC_CreateHandle(
        &(HikVision::handle), 
        stDeviceList.pDeviceInfo[nIndex]
    );

    // 打开设备
    HikVision::nRet = MV_CC_OpenDevice(HikVision::handle);

    // 设置触发模式为off
    HikVision::nRet = MV_CC_SetEnumValue(HikVision::handle, "TriggerMode", 0);

    // 获取像素格式
    MVCC_ENUMVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_ENUMVALUE));
    nRet = MV_CC_GetPixelFormat(handle, &stParam);
    if (MV_OK == nRet) {
        std::cout << "Current PixelFormat: " << stParam.nCurValue << std::endl;
        // 这里需要根据返回的数值转化为16进制在Pixelype.h查找对应的像素格式名称
    } else {
        std::cout << "Get PixelFormat fail!" << std::endl;
    }

    return true;
}

void HikVision::setCameraParams() {
    std::string yaml_file_path = 
        (fs::path(__FILE__).parent_path() / "../../config/camera-params.yaml").string(); // yaml文件路径
    // 设置宽度
    int Width = std::stoi(getParam("Width", yaml_file_path));
    HikVision::nRet = MV_CC_SetIntValue(HikVision::handle, "Width", Width);
    HikVision::stImageInfo.nWidth = Width;
    
    // 设置高度
    int Height = std::stoi(getParam("Height", yaml_file_path));
    HikVision::nRet = MV_CC_SetIntValue(HikVision::handle, "Height", Height);
    HikVision::stImageInfo.nHeight = Height;

    // 设置x偏移
    int OffsetX = std::stoi(getParam("OffsetX", yaml_file_path));
    HikVision::nRet = MV_CC_SetIntValue(HikVision::handle, "OffsetX", OffsetX);
    HikVision::stImageInfo.nOffsetX = OffsetX;

    // 设置y偏移
    int OffsetY = std::stoi(getParam("OffsetY", yaml_file_path));
    HikVision::nRet = MV_CC_SetIntValue(HikVision::handle, "OffsetY", OffsetY);
    HikVision::stImageInfo.nOffsetY = OffsetY;

    // 设置曝光时间
    float ExposureTime = std::stof(getParam("ExposureTime", yaml_file_path)); 
    HikVision::nRet = MV_CC_SetFloatValue(HikVision::handle, "ExposureTime", ExposureTime);
    HikVision::stImageInfo.fExposureTime = ExposureTime;

    // 设置增益
    float Gain = std::stof(getParam("Gain", yaml_file_path)); 
    HikVision::nRet = MV_CC_SetFloatValue(HikVision::handle, "Gain", Gain);
    HikVision::stImageInfo.fGain = Gain;

}

bool HikVision::releaseCamera() {
    // 关闭设备
    nRet = MV_CC_CloseDevice(handle);

    // 销毁句柄
    nRet = MV_CC_DestroyHandle(handle);

    handle = nullptr;

    // 反初始化SDK
    MV_CC_Finalize();
    return true;
}

} // namespace camera
