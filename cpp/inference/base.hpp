#pragma once

#include <string>
#include <vector>
#include <map>

class InferenceEngineBase {
public:
    virtual int init(const std::string &model_path) = 0;
    virtual void preprocess() = 0;
    virtual std::map<std::string, std::vector<float>> forward(void *data, size_t size) = 0;
    virtual void postprocess() = 0;

    virtual ~InferenceEngineBase() = default;
};