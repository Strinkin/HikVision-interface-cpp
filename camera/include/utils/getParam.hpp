/**
 * @author: strinkin
 * @date: 2024-8-16
 * 从yaml文件获取参数
*/
#ifndef __GETPARAM_HPP__
#define __GETPARAM_HPP__
//STD
#include <iostream>
#include <string>
#include <filesystem>
//3rd parted
#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

inline std::string getParam(const std::string& param_name, const std::string& yaml_file_path) 
{
    std::string param_value;
    try {
        // std::cout << __FILE__ << ": Current working directory: " << fs::current_path().string() << std::endl;
        // 读取数据
        // 加载YAML文件
        YAML::Node config = YAML::LoadFile(yaml_file_path);
        std::cout << "load yaml file from " + yaml_file_path << std::endl;
        param_value = config[param_name].as<std::string>();
        std::cout << param_name + ": " + param_value << std::endl; 
        return param_value;
    } catch (YAML::Exception& e) {
        // 处理异常，比如文件加载失败或者参数不存在等
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        return "";
    }
}

// 重载函数，用于获取数组类型的参数
#include <vector>
#include <type_traits>

template<typename T>
inline T getParam(const std::string& param_name, const std::string& yaml_file_path, const std::string& param_type)
{
    std::vector<double> param_values;
    cv::Scalar threshold;

    try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);
        std::cout << "Loaded YAML file from " + yaml_file_path << std::endl;

        if (param_type == "array") 
        {
            const YAML::Node& node = config[param_name];
            if (node.IsSequence()) 
            {
                for (const auto& item : node) 
                {
                    param_values.push_back(item.as<double>());
                }
            } 
            else 
            {
                std::cerr << "Expected an array for " << param_name << " but found a different type." << std::endl;
            }

            for (int i = 0; i < param_values.size(); ++i) 
            {
                threshold[i] = param_values[i];
                printf("%s: %f\n", param_name.c_str(), param_values[i]);
            }

            if constexpr (std::is_same_v<T, cv::Scalar>) {
                return threshold; // Return cv::Scalar for "array"
            }
        } 
        else if (param_type == "camera_matrix") 
        {
            const YAML::Node& node = config[param_name]["data"];
            if (node.IsSequence()) 
            {
                for (const auto& item : node) 
                {
                    param_values.push_back(item.as<double>());
                }
            }
            else 
            {
                std::cerr << "Expected an array for " << param_name << " but found a different type." << std::endl;
            }

            if constexpr (std::is_same_v<T, std::vector<double>>) {
                return param_values; // Return vector<double> for "camera_matrix"
            }
        } 
        else if (param_type == "dist_coeffs") 
        {
            const YAML::Node& node = config[param_name]["data"];
            if (node.IsSequence()) 
            {
                for (const auto& item : node) 
                {
                    param_values.push_back(item.as<double>());
                }
            }
            else 
            {
                std::cerr << "Expected an array for " << param_name << " but found a different type." << std::endl;
            }

            if constexpr (std::is_same_v<T, std::vector<double>>) {
                return param_values; // Return vector<double> for "dist_coeffs"
            }
        } 
        else 
        {
            std::cerr << "Invalid param_type specified: " << param_type << std::endl;
        }
    } catch (YAML::Exception& e) {
        std::cerr << "Error reading YAML file: " << e.what() << std::endl;
    }

    return T{}; // Return a default-constructed T if no valid data is found
}

#endif // __GETPARAM_HPP__