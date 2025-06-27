# HikVision-interface-cpp
海康威视c++二次开发
# 环境要求
`yaml-cpp`
`OpenCV4`
`C++17`
# 使用说明
- 执行以下命令下载yaml-cpp到本地
```shell
sudo apt install libyaml-cpp-dev
```
- 克隆仓库到本地
- 进入example目录
- 执行命令
```shell
mkdir build && cd build && cmake .. && make && cd example/bin && ./example
```
### 示例代码在example/example.cpp中, 使用CS016-10UC进行测试, 暂不支持网口设备, 如有问题请联系strinkin@qq.com
### 代码能力有限, 如代码出现bug请见谅, 可以联系本人进行修复

# TODO:
- [ ] 使用运行时更高效CRTP重构代码
