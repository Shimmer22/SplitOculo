#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

// Base64 encoding helper
static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

// Preprocessing: Resize, CenterCrop, Normalize
cv::Mat preprocess(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
    
    // Center crop 224x224
    int x = (resized.cols - 224) / 2;
    int y = (resized.rows - 224) / 2;
    cv::Rect crop_region(x, y, 224, 224);
    cv::Mat cropped = resized(crop_region);

    cv::Mat float_img;
    cropped.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    // Normalize
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::Mat normalized;
    cv::subtract(float_img, mean, normalized);
    cv::divide(normalized, std, normalized);
    
    return normalized;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [server_url]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string server_url = (argc > 3) ? argv[3] : "http://localhost:8080";

    // 1. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    
    // OpenCV loads BGR, convert to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    auto start_encode = std::chrono::high_resolution_clock::now();
    
    cv::Mat preprocessed = preprocess(img);
    
    // CHW format for ONNX
    cv::Mat blob = cv::dnn::blobFromImage(preprocessed); // 1, 3, 224, 224

    // 2. ONNX Inference
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SplitOculoEdge");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    Ort::Session session(env, model_path.c_str(), session_options);

    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
    
    // Create input tensor
    std::array<int64_t, 4> input_shape = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::memcpy(input_tensor_values.data(), blob.ptr<float>(), input_tensor_size * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    
    // 3. Post-processing (Quantization)
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_count = type_info.GetElementCount();
    
    // Find min/max
    float f_min = floatarr[0];
    float f_max = floatarr[0];
    for(size_t i=0; i<output_count; ++i) {
        if(floatarr[i] < f_min) f_min = floatarr[i];
        if(floatarr[i] > f_max) f_max = floatarr[i];
    }
    
    float scale = (f_max - f_min) / 255.0f;
    float zero_point = -f_min / scale;
    
    std::vector<uint8_t> quantized(output_count);
    for(size_t i=0; i<output_count; ++i) {
        float val = floatarr[i] / scale + zero_point;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        quantized[i] = static_cast<uint8_t>(std::round(val));
    }
    
    auto end_encode = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(end_encode - start_encode).count();
    
    // 4. Send to Server
    std::string b64 = base64_encode(quantized.data(), quantized.size());
    
    nlohmann::json payload;
    payload["features"] = b64;
    payload["scale"] = scale;
    payload["zero_point"] = zero_point;
    payload["prompt"] = "What is in this image?";
    
    std::cout << "Encoded in " << encode_ms << " ms. Payload size: " << b64.size() << " bytes." << std::endl;
    std::cout << "Sending to " << server_url << "..." << std::endl;
    
    httplib::Client cli(server_url.c_str());
    cli.set_connection_timeout(5);
    cli.set_read_timeout(300); // 5 min
    
    auto res = cli.Post("/infer", payload.dump(), "application/json");
    
    if (res && res->status == 200) {
        auto result = nlohmann::json::parse(res->body);
        std::cout << "Response:\n" << result["response"].get<std::string>() << std::endl;
    } else {
        std::cerr << "Request failed. Code: " << (res ? res->status : 0) << std::endl;
    }
    
    return 0;
}
