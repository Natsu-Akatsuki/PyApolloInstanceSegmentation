# PyApolloInstanceSegmentation

- CNN方案，支持**16线机械式激光雷达**，**需显卡**
- 面向具有深度学习环境配置经验的同学（非草履虫文档，尚在完善）
- 提供`Apollo`实例分割（CNN）的`Python`封装（基于`pybind11`），便于进行算法开发和算法验证。具体实现可参考`script`模块的内容

```python
# input：点云
# output：label mask（-1：背景点，正数：label）
lidar_apollo_instance_segmentation = LidarApolloInstanceSegmentation()
mask = lidar_apollo_instance_segmentation.segmentation(pointcloud)
```

- 感谢百度`Apollo`的开源的16线模型，`Autoware`提供的`TensorRT`封装

## Demo

去行人（基于ROS）：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/instance_segmentation.gif" alt="instance_segmentation" style="zoom:67%;" />

去点云地图拖影（基于ROS）：

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20221204163514816.png" alt="image-20221204163514816" style="zoom: 67%;" />

## Usage

### Requirement

- 深度学习依赖（`TensorRT`的版本估摸需要8+）

|     依赖      |     已测试的版本     |
| :-----------: | :------------------: |
| nvidia driver |          —           |
|   TensorRT    | 8.4.1.5（tar包安装） |
|     cuda      |      11.5.r11.5      |
|     cudnn     |        8.2.4         |

- apt package

```bash
# 输入格式化依赖库
$ sudo apt install libfmt-dev
```

- Python点云工具库，[ampcl](https://github.com/Natsu-Akatsuki/PointCloud-PyUsage)

```bash
$ git clone https://github.com/Natsu-Akatsuki/PointCloud-PyUsage --depth=1
$ cd PointCloud-PyUsage
$ bash install.sh
```

- 下载相对应的**caffe**模型

```bash
$ cd data

# For VLP-16
$ wget -c https://github.com/ApolloAuto/apollo/raw/88bfa5a1acbd20092963d6057f3a922f3939a183/modules/perception/production/data/perception/lidar/models/cnnseg/velodyne16/deploy.caffemodel

# For HDL-64
$ wget -c https://github.com/ApolloAuto/apollo/raw/88bfa5a1acbd20092963d6057f3a922f3939a183/modules/perception/production/data/perception/lidar/models/cnnseg/velodyne64/deploy.caffemodel

# For VLS-128 
$ wget -c https://github.com/ApolloAuto/apollo/raw/91844c80ee4bd0cc838b4de4c625852363c258b5/modules/perception/production/data/perception/lidar/models/cnnseg/velodyne128/deploy.caffemodel 
```

注意事项：需导入各种环境变量到`~/.bashrc`

```bash
# example
export PATH="/home/helios/.local/bin:$PATH"
CUDA_PATH=/usr/local/cuda/bin
TENSORRT_PATH=${HOME}/Application/TensorRT-8.4.1.5/bin
CUDA_LIB_PATH=/usr/local/cuda/lib64
TENSORRT_LIB_PATH=${HOME}/Application/TensorRT-8.4.1.5/lib
PYTORCH_LIB_PATH=${HOME}/application/libtorch/lib
export PATH=${PATH}:${CUDA_PATH}:${TENSORRT_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_LIB_PATH}:${TENSORRT_LIB_PATH}:${PYTORCH_LIB_PATH}
```

### Install and Build

- 修改`CMakeLists`：修改其中的`TensorRT`等依赖库的路径

```bash
# 编译Python拓展库
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```

- 执行例程

```bash
$ cd script
$ python3 single_frame_demo.py
```

![image-20221205145251883](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20221205145251883.png)

## Inner-workings / Algorithms

See the [original design](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/3d_obstacle_perception.md) by Apollo.

## Parameters

### Core Parameters

|          Name           |  Type  |    Default Value     |                         Description                          |
| :---------------------: | :----: | :------------------: | :----------------------------------------------------------: |
|    `score_threshold`    | double |         0.8          | If the score of a detected object is lower than this value, the object is ignored. |
|         `range`         |  int   |          60          |         Half of the length of feature map sides. [m]         |
|         `width`         |  int   |         640          |                The grid width of feature map.                |
|        `height`         |  int   |         640          |               The grid height of feature map.                |
|      `engine_file`      | string |   "vls-128.engine"   |       The name of TensorRT engine file for CNN model.        |
|     `prototxt_file`     | string |  "vls-128.prototxt"  |           The name of prototxt file for CNN model.           |
|    `caffemodel_file`    | string | "vls-128.caffemodel" |          The name of caffemodel file for CNN model.          |
| `use_intensity_feature` |  bool  |         true         |       The flag to use intensity feature of pointcloud.       |
| `use_constant_feature`  |  bool  |        false         | The flag to use direction and distance feature of pointcloud. |
|     `target_frame`      | string |     "base_link"      |       Pointcloud data is transformed into this frame.        |
|       `z_offset`        |  int   |          2           |               z offset from target frame. [m]                |

## Assumptions / Known limits

There is no training code for CNN model.

### Reference

- [Autoware.Universe Implementation](https://github.com/autowarefoundation/autoware.universe/tree/main/perception/lidar_apollo_instance_segmentation)

