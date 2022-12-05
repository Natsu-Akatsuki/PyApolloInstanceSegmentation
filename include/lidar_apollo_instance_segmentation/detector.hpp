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

#pragma once

#include "cluster2d.hpp"
#include "feature_generator.hpp"

#include <TrtNet.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fmt/color.h>
#include <memory>
#include <string>

using namespace pybind11::literals;

class LidarApolloInstanceSegmentation {
public:
  explicit LidarApolloInstanceSegmentation();
  ~LidarApolloInstanceSegmentation() {}
  std::vector<int> detectDynamicObjects(const pybind11::array_t<float> &input);

private:
  std::unique_ptr<Tn::trtNet> net_ptr_;
  std::shared_ptr<Cluster2D> cluster2d_;
  std::shared_ptr<FeatureGenerator> feature_generator_;
  float score_threshold_;
};
