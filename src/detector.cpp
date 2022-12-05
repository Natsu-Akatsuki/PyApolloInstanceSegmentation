// Copyright 2020 TierIV
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lidar_apollo_instance_segmentation/detector.hpp"

#include "lidar_apollo_instance_segmentation/feature_map.hpp"

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <filesystem>
#include "yaml-cpp/yaml.h"
#include <pybind11/embed.h>
LidarApolloInstanceSegmentation::LidarApolloInstanceSegmentation() {
  int range, width, height;
  bool use_intensity_feature, use_constant_feature;
  std::string engine_file;
  std::string prototxt_file;
  std::string caffemodel_file;

  std::filesystem::path file_path(__FILE__);
  auto dir_path = file_path.parent_path().parent_path();
  auto config_path = std::string(dir_path / "config" / "vlp-16.param.yaml");
  fmt::print(fg(fmt::color::green), "[INFO] load config file {}.\n", config_path);
  YAML::Node config = YAML::LoadFile(config_path);
  score_threshold_ = config["score_threshold"].as<float>();
  range = config["range"].as<int>();
  width = config["width"].as<int>();
  height = config["height"].as<int>();
  use_intensity_feature = config["use_intensity_feature"].as<bool>();
  use_constant_feature = config["use_constant_feature"].as<bool>();
  engine_file = config["engine_file"].as<std::string>();
  prototxt_file = config["prototxt_file"].as<std::string>();
  caffemodel_file = config["caffemodel_file"].as<std::string>();

  engine_file = std::string(dir_path / engine_file);
  prototxt_file = std::string(dir_path / prototxt_file);
  caffemodel_file = std::string(dir_path / caffemodel_file);

  // load weight file
  std::ifstream fs(engine_file);
  if (!fs.is_open()) {
    fmt::print(fg(fmt::color::red),
               "[ERROR] Could not find {}. try making TensorRT engine from caffemodel and prototxt\n",
               engine_file);

    Tn::Logger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    const nvcaffeparser1::IBlobNameToTensor *blob_name2tensor = parser->parse(
        prototxt_file.c_str(), caffemodel_file.c_str(), *network, nvinfer1::DataType::kFLOAT);
    std::string output_node = "deconv0";
    auto output = blob_name2tensor->find(output_node.c_str());
    if (output == nullptr) {
      fmt::print(fg(fmt::color::red), "[ERROR] can not find output named {}\n", output_node);
    }
    network->markOutput(*output);
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
#else
    config->setMaxWorkspaceSize(1 << 30);
#endif
    nvinfer1::IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
    assert(plan != nullptr);
    std::ofstream outfile(engine_file, std::ofstream::binary);
    assert(!outfile.fail());
    outfile.write(reinterpret_cast<char *>(plan->data()), plan->size());
    outfile.close();
    if (network) {
      delete network;
    }
    if (parser) {
      delete parser;
    }
    if (builder) {
      delete builder;
    }
    if (config) {
      delete config;
    }
    if (plan) {
      delete plan;
    }
  }
  net_ptr_.reset(new Tn::trtNet(engine_file));

  // feature map generator: pre process
  feature_generator_ = std::make_shared<FeatureGenerator>(
      width, height, range, use_intensity_feature, use_constant_feature);

  // cluster: post process
  cluster2d_ = std::make_shared<Cluster2D>(width, height, range);
}

std::vector<int> LidarApolloInstanceSegmentation::detectDynamicObjects(const pybind11::array_t<float> &input) {
  // convert from py::array to pcl
  auto input_ref = input.unchecked<2>();
  pcl::PointCloud<pcl::PointXYZI>::Ptr
      pcl_pointcloud_raw_ptr(new pcl::PointCloud<pcl::PointXYZI>(input_ref.shape(0), 1));
  for (int i = 0; i < input_ref.shape(0); ++i) {
    pcl_pointcloud_raw_ptr->points[i].x = input_ref(i, 0);
    pcl_pointcloud_raw_ptr->points[i].y = input_ref(i, 1);
    pcl_pointcloud_raw_ptr->points[i].z = input_ref(i, 2);
    pcl_pointcloud_raw_ptr->points[i].intensity = input_ref(i, 3);
  }

  // generate feature map
  std::shared_ptr<FeatureMapInterface> feature_map_ptr =
      feature_generator_->generate(pcl_pointcloud_raw_ptr);

  // inference
  std::shared_ptr<float> inferred_data(new float[net_ptr_->getOutputSize() / sizeof(float)]);
  net_ptr_->doInference(feature_map_ptr->map_data.data(), inferred_data.get());

  // post process
  const float objectness_thresh = 0.5;
  pcl::PointIndices valid_idx;
  valid_idx.indices.resize(pcl_pointcloud_raw_ptr->size());
  std::iota(valid_idx.indices.begin(), valid_idx.indices.end(), 0);
  cluster2d_->cluster(
      inferred_data, pcl_pointcloud_raw_ptr, valid_idx, objectness_thresh,
      true /*use all grids for clustering*/);
  const float height_thresh = 0.5;
  const int min_pts_num = 3;
  std::vector<int> output;
  cluster2d_->getObjects(
      score_threshold_, height_thresh, min_pts_num, output);

  return output;
}

// int main(int argc, char **argv) {
//   pybind11::scoped_interpreter guard{};
//   LidarApolloInstanceSegmentation lidar_apollo_instance_segmentation;
//   lidar_apollo_instance_segmentation.detectDynamicObjects();
// }

PYBIND11_MODULE(lidar_apollo_instance_segmentation_pyb, m) {
  m.doc() = "pybind11 for apollo instance segmentation";
  pybind11::class_<LidarApolloInstanceSegmentation>(m, "LidarApolloInstanceSegmentation")
      .def(pybind11::init<>())
      .def("segmentation",
           &LidarApolloInstanceSegmentation::detectDynamicObjects,
           "apollo instance segmentation", pybind11::arg("input"));
}