#ifndef TWINLITENET_ONNXRUNTIME_HPP_
#define TWINLITENET_ONNXRUNTIME_HPP_

#include <opencv2/opencv.hpp>
#include <numeric>
#include "onnxruntime_float16.h"
#include "onnxruntime_cxx_api.h"

class TwinLiteNet
{
public:
        TwinLiteNet(std::string model_path, int cuda_device_id = 0);
        ~TwinLiteNet();
        void Infer(const cv::Mat &image, cv::Mat &da_out, cv::Mat &ll_out);

private:
        Ort::Env env_;
        Ort::Session *session_;
        Ort::SessionOptions session_options_;
        Ort::RunOptions run_options_;
        Ort::AllocatorWithDefaultOptions allocator_;

#ifdef ENABLE_CUDA
        OrtCUDAProviderOptions cuda_option_;
#endif
        std::vector<const char *> input_node_names_;
        std::vector<const char *> output_node_names_;
        std::vector<int64_t> input_node_dims_ = {1, 3, 360, 640};
        std::vector<int> per_outsection_dims = {2, 360, 640}; // pls refer twinlitenet architecture
};

#endif // TWINLITENET_ONNXRUNTIME_HPP_