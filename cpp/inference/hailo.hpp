#pragma once

#include "base.hpp"
#include <hailo/hailort.hpp>
#include <hailo/vstream.hpp>
#include <hailo/vdevice.hpp>

using namespace hailort;

class HailoInferenceEngine : InferenceEngineBase
{
private:
    std::unique_ptr<VDevice> m_vdevice;
    std::unique_ptr<ConfiguredNetworkGroup> m_network_group;
    std::unique_ptr<InputVStream> m_input_vstream;
    std::unique_ptr<std::vector<OutputVStream>> m_output_vstreams;

public:
    int init(const std::string &model_path) override;
    void preprocess() override;
    std::map<std::string, std::vector<float>> forward(void *data, size_t size) override;
    void postprocess() override;

    ~HailoInferenceEngine() override;
};