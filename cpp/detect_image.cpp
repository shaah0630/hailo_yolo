/**
 * detect_image.cpp
 * 
 * Single image detection with Hailo-8L + Post-processing in C++.
 * Usage: ./detect_image <image_path> [hef_path] [conf_threshold]
 */
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <map>

#include <opencv2/opencv.hpp>
#include <hailo/hailort.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vstream.hpp>

#include "hailo.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"

using namespace hailort;

// Constants
constexpr int TARGET_WIDTH = 640;
constexpr int TARGET_HEIGHT = 640;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./detect_image <image_path> [model_path] [conf_threshold]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "../models/yolo26n.hef";
    float conf_threshold = (argc > 3) ? std::stof(argv[3]) : 0.25f;

    std::cout << "[Loading image: " << image_path << "]" << std::endl;

    // 1. Preprocess Image (Letterbox)
    cv::Mat orig_image = cv::imread(image_path);
    if (orig_image.empty()) {
        std::cerr << "Error: Could not read image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "✓ Original image size: " << orig_image.cols << "x" << orig_image.rows << std::endl;

    LetterboxInfo lb_info = letterbox_image(orig_image, TARGET_WIDTH, TARGET_HEIGHT);
    cv::Mat resized_image = lb_info.image;
    // Convert color space from BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    std::cout << "✓ Preprocessed to: (" << resized_image.rows << ", " << resized_image.cols << ", " << resized_image.channels() << ")" << std::endl;

    // 2. Setup inference engine
    auto inf_eng = std::make_unique<HailoInferenceEngine>();
    if (inf_eng->init(model_path) != 0) {
        std::cout << "Failed to initialize inference engine: " << "!" << std::endl;
        return -1;
    }

    // 3. Inference
    std::cout << "[Running inference...]" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::map<std::string, std::vector<float>> output_data = inf_eng->forward(resized_image.data, resized_image.total() * resized_image.elemSize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "✓ Inference completed in " << duration.count() << "ms" << std::endl;

    // 4. Map outputs to expected structure for post-processing
    std::vector<const float*> cls_ptrs;
    std::vector<const float*> reg_ptrs;
    
    if (!map_output_tensors(output_data, cls_ptrs, reg_ptrs)) {
        std::cerr << "Error: Could not identify all required output tensors by size." << std::endl;
        std::cerr << "Sizes found:" << std::endl;
        for (auto& pair : output_data) std::cout << pair.first << ": " << pair.second.size() << std::endl;
        return 1;
    }

    // 5. Post Process
    std::vector<Detection> detections = run_postprocess(
        IntList<8, 16, 32>{},
        IntList<80, 40, 20>{},
        cls_ptrs,
        reg_ptrs,
        conf_threshold
    );

    std::cout << "✓ Found " << detections.size() << " detections above threshold " << conf_threshold << std::endl;

    // 6. Visualize
    for (const auto& det : detections) {
        // Reverse Letterbox: (x - pad) / scale
        float x1_f = (det.x1 - lb_info.pad_w) / lb_info.scale;
        float y1_f = (det.y1 - lb_info.pad_h) / lb_info.scale;
        float x2_f = (det.x2 - lb_info.pad_w) / lb_info.scale;
        float y2_f = (det.y2 - lb_info.pad_h) / lb_info.scale;
        
        // Clip
        int x1 = std::max(0, std::min((int)x1_f, orig_image.cols));
        int y1 = std::max(0, std::min((int)y1_f, orig_image.rows));
        int x2 = std::max(0, std::min((int)x2_f, orig_image.cols));
        int y2 = std::max(0, std::min((int)y2_f, orig_image.rows));
        
        cv::rectangle(orig_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        std::string label = std::string(det.cls_name) + " " + std::to_string(det.conf).substr(0, 4);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(orig_image, cv::Point(x1, y1 - textSize.height - 5), cv::Point(x1 + textSize.width, y1), cv::Scalar(0, 255, 0), -1);
        cv::putText(orig_image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        std::cout << "  " << label << " at [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
    }

    std::string output_filename = "output_detected.jpg";
    cv::imwrite(output_filename, orig_image);
    std::cout << "✓ Output image saved to: " << output_filename << std::endl;

    return 0;
}
