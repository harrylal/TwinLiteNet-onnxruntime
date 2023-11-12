#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "twinlitenet_onnxruntime.hpp"

int main()
{
    // Model Loading and associated initializations
    TwinLiteNet twinlitenet("/mnt/sdd1/personal/TwinLiteNet-onnxruntime/models/best.onnx");
    std::cout << "Model Loaded" << std::endl;

    // Directory containing images
    std::string directory_path = "/mnt/sdd1/personal/TwinLiteNet-onnxruntime/images/";

    // Get a list of all image files in the directory
    std::vector<cv::String> image_files;
    cv::glob(directory_path + "*.jpg", image_files);

    // Iterate through each image file
    for (int i = 0; i < image_files.size(); ++i)
    {
        // Read the image
        cv::Mat img = cv::imread(image_files[i]);

        if (img.empty())
        {
            std::cerr << "Error: Could not read the image " << image_files[i] << std::endl;
            continue; // Skip to the next iteration if the image cannot be read
        }

        // Resize the image to 360x640 as the model expects this size
        cv::resize(img, img, cv::Size(640, 360));
        cv::Mat img_vis = img.clone();

        cv::Mat da_out, ll_out;

        // Run the inference
        twinlitenet.Infer(img, da_out, ll_out);
        std::cout << "Inference Done for " << image_files[i] << std::endl;

        // Plot the predictions
        img_vis.setTo(cv::Scalar(255, 0, 0), da_out);   // Set drivable area to red
        img_vis.setTo(cv::Scalar(0, 255, 255), ll_out); // Set road lanes to cyan

        // Save the image
        std::string result_path = "results" + std::to_string(i) + ".jpg";
        cv::imwrite(result_path, img_vis);
        std::cout << "Results saved to " << result_path << std::endl;
    }

    return 0;
}
