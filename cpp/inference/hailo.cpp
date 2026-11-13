#include "hailo.hpp"

#include <iostream>

// Helper to print errors
void print_error(hailo_status status, const std::string &msg)
{
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

int HailoInferenceEngine::init(const std::string &model_path)
{
    // Setup Hailo
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        print_error(vdevice_exp.status(), "Failed to create VDevice");
        return -1;
    }
    m_vdevice = std::move(vdevice_exp.value());

    // Configure network group
    auto hef_exp = Hef::create(model_path);
    if (!hef_exp) {
        print_error(hef_exp.status(), "Failed to load HEF");
        return -2;
    }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = m_vdevice->create_configure_params(hef);
    if (!configure_params_exp) {
        print_error(configure_params_exp.status(), "Failed to create configure params");
        return -3;
    }
    auto configure_params = configure_params_exp.value();

    auto network_groups_exp = m_vdevice->configure(hef, configure_params);
    if (!network_groups_exp) {
        print_error(network_groups_exp.status(), "Failed to configure network group");
        return -4;
    }
    auto network_groups = std::move(network_groups_exp.value());
    auto m_network_group = network_groups[0];

    // Create VStreams
    // Input: UINT8, NHWC/Auto (Matches image)
    // Output: FLOAT32 (Matches requirement for automatic dequantization)
    // Let's create params manually to force FLOAT32 on output
    auto input_vstream_params = m_network_group->make_input_vstream_params(false,
                                                                            HAILO_FORMAT_TYPE_UINT8,
                                                                            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                            HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstream_params) {
        print_error(input_vstream_params.status(), "Failed to make input params");
        return -5;
    }
    
    auto output_vstream_params = m_network_group->make_output_vstream_params(false,
                                                                             HAILO_FORMAT_TYPE_FLOAT32,
                                                                             HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                             HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstream_params) {
        print_error(output_vstream_params.status(), "Failed to make output params");
        return -6;
    }

    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*m_network_group, input_vstream_params.value());
    if (!input_vstreams_exp) {
        print_error(input_vstreams_exp.status(), "Failed to create input vstreams");
        return -7;
    }
    auto input_vstreams = std::move(input_vstreams_exp.value());
    // Assuming single input
    if (input_vstreams.size() != 1) {
        std::cerr << "Error: Expected 1 input stream, got " << input_vstreams.size() << std::endl;
        return -8;
    }
    m_input_vstream = std::make_unique<InputVStream>(std::move(input_vstreams[0]));

    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*m_network_group, output_vstream_params.value());
    if (!output_vstreams_exp) {
        print_error(output_vstreams_exp.status(), "Failed to create output vstreams");
        return -9;
    }
    m_output_vstreams = std::make_unique<std::vector<OutputVStream>>(output_vstreams_exp.release());

    return 0;
}

void HailoInferenceEngine::preprocess()
{
}

std::map<std::string, std::vector<float>>
HailoInferenceEngine::forward(void *data, size_t size)
{
    std::map<std::string, std::vector<float>> output_data;

    /* Send input */
    hailo_status status = m_input_vstream->write(MemoryView(data, size));
    if (status != HAILO_SUCCESS) {
        print_error(status, "Failed to write input");
        return output_data;
    }

    /* Read outputs */
    // We expect cls_80, cls_40, cls_20, and reg.
    // We need to map them by name.

    for (auto &output_vstream : *m_output_vstreams) {
        std::string name = output_vstream.name(); // This is the vstream name

        // The VStream names might be slightly different from the python dictionary keys if python remapped them.
        // In common.py: shape_to_name map was used for Python...
        // But here we might get raw names from HEF.
        // We will read the data and try to identify by shape later or hope names make sense.

        size_t framesize = output_vstream.get_frame_size();
        std::vector<float> buffer(framesize / sizeof(float)); // Since we requested FLOAT32

        status = output_vstream.read(MemoryView(buffer.data(), framesize));
        if (status != HAILO_SUCCESS) {
            print_error(status, "Failed to read output " + name);
            return output_data;
        }

        output_data[name] = std::move(buffer);
    }

    return output_data;
}

void HailoInferenceEngine::postprocess()
{
}

HailoInferenceEngine::~HailoInferenceEngine()
{
}
