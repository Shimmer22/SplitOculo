#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    ret.reserve(((in_len + 2) / 3) * 4);
    
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

            for(i = 0; i < 4; i++)
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

        for (j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];

        while(i++ < 3)
            ret += '=';
    }

    return ret;
}

class EdgeClient {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    Ort::MemoryInfo memory_info_;
    
    static constexpr float mean_[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float std_[3] = {0.229f, 0.224f, 0.225f};
    
public:
    EdgeClient(const std::string& model_path, int num_threads = 4) 
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SplitOculoEdge");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();
        
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_names_.push_back(strdup(input_name.get()));
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_names_.push_back(strdup(output_name.get()));
    }
    
    std::pair<std::vector<uint8_t>, std::pair<float, float>> 
    infer_and_quantize(const cv::Mat& img) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        cv::Mat preprocessed = preprocess(img);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        std::array<int64_t, 4> input_shape = {1, 3, 224, 224};
        size_t input_tensor_size = 1 * 3 * 224 * 224;
        
        std::vector<float> input_tensor_values(input_tensor_size);
        std::memcpy(input_tensor_values.data(), preprocessed.ptr<float>(), input_tensor_size * sizeof(float));
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values.data(), input_tensor_size, 
            input_shape.data(), input_shape.size());
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), &input_tensor, 1, 
            output_names_.data(), 1);
        
        auto t3 = std::chrono::high_resolution_clock::now();
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_count = type_info.GetElementCount();
        
        auto [quantized, scale, zero_point] = quantize(output_data, output_count);
        
        auto t4 = std::chrono::high_resolution_clock::now();
        
        double preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double tensor_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double quant_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
        
        std::cout << "[Timing] Preprocess: " << preprocess_ms << " ms, "
                  << "Tensor prep: " << tensor_ms << " ms, "
                  << "Inference: " << infer_ms << " ms, "
                  << "Quantize: " << quant_ms << " ms" << std::endl;
        
        return {quantized, {scale, zero_point}};
    }
    
    void warmup(int iterations = 3) {
        std::vector<float> dummy_input(3 * 224 * 224, 0.5f);
        std::array<int64_t, 4> input_shape = {1, 3, 224, 224};
        
        for (int i = 0; i < iterations; i++) {
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, dummy_input.data(), dummy_input.size(),
                input_shape.data(), input_shape.size());
            session_->Run(Ort::RunOptions{nullptr}, 
                         input_names_.data(), &input_tensor, 1,
                         output_names_.data(), 1);
        }
        std::cout << "[Warmup] Completed " << iterations << " warmup iterations" << std::endl;
    }
    
private:
    cv::Mat preprocess(const cv::Mat& img) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        
        int x = (resized.cols - 224) / 2;
        int y = (resized.rows - 224) / 2;
        cv::Rect crop_region(x, y, 224, 224);
        cv::Mat cropped = resized(crop_region);
        
        cv::Mat float_img;
        cropped.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);
        
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        
        for (int c = 0; c < 3; c++) {
            channels[c] = (channels[c] - mean_[c]) / std_[c];
        }
        
        cv::Mat normalized;
        cv::merge(channels, normalized);
        
        return normalized;
    }
    
    std::tuple<std::vector<uint8_t>, float, float> 
    quantize(const float* data, size_t count) {
        float f_min = data[0];
        float f_max = data[0];
        
        for (size_t i = 1; i < count; i++) {
            if (data[i] < f_min) f_min = data[i];
            if (data[i] > f_max) f_max = data[i];
        }
        
        float scale = (f_max - f_min) / 255.0f;
        float zero_point = -f_min / scale;
        
        std::vector<uint8_t> quantized(count);
        
        #pragma omp parallel for
        for (size_t i = 0; i < count; i++) {
            float val = data[i] / scale + zero_point;
            quantized[i] = static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
        }
        
        return {quantized, scale, zero_point};
    }
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model_path> <image_path> [server_url] [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --threads N     Set inference threads (default: 4)" << std::endl;
    std::cerr << "  --warmup N      Run N warmup iterations (default: 3)" << std::endl;
    std::cerr << "  --benchmark     Run benchmark mode (10 iterations)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string server_url = (argc > 3 && argv[3][0] != '-') ? argv[3] : "http://localhost:8080";
    
    int num_threads = 4;
    int warmup_iters = 3;
    bool benchmark_mode = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iters = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark_mode = true;
        }
    }
    
    std::cout << "============================================================" << std::endl;
    std::cout << "SplitOculo C++ Edge Client (Optimized)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Server: " << server_url << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << std::endl;
    
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    
    std::cout << "[Init] Loading model..." << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    
    EdgeClient client(model_path, num_threads);
    
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "[Init] Model loaded in " << load_ms << " ms" << std::endl;
    
    if (warmup_iters > 0) {
        client.warmup(warmup_iters);
    }
    
    if (benchmark_mode) {
        std::cout << std::endl;
        std::cout << "[Benchmark] Running 10 iterations..." << std::endl;
        
        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto [quantized, params] = client.infer_and_quantize(img_rgb);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        
        double total = std::accumulate(times.begin(), times.end(), 0.0);
        double avg = total / times.size();
        double min_t = *std::min_element(times.begin(), times.end());
        double max_t = *std::max_element(times.begin(), times.end());
        
        std::cout << "[Benchmark] Avg: " << avg << " ms, "
                  << "Min: " << min_t << " ms, "
                  << "Max: " << max_t << " ms" << std::endl;
        return 0;
    }
    
    auto start_encode = std::chrono::high_resolution_clock::now();
    
    auto [quantized, params] = client.infer_and_quantize(img_rgb);
    float scale = params.first;
    float zero_point = params.second;
    
    auto end_encode = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(end_encode - start_encode).count();
    
    std::string b64 = base64_encode(quantized.data(), quantized.size());
    
    nlohmann::json payload;
    payload["features"] = b64;
    payload["scale"] = scale;
    payload["zero_point"] = zero_point;
    payload["prompt"] = "What is in this image?";
    
    std::cout << std::endl;
    std::cout << "[Summary] Encode time: " << encode_ms << " ms" << std::endl;
    std::cout << "[Summary] Payload size: " << b64.size() << " bytes (" 
              << b64.size() / 1024.0 << " KB)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "[Network] Sending to " << server_url << "..." << std::endl;
    
    httplib::Client cli(server_url.c_str());
    cli.set_connection_timeout(5);
    cli.set_read_timeout(300);
    
    auto t_network_start = std::chrono::high_resolution_clock::now();
    auto res = cli.Post("/infer", payload.dump(), "application/json");
    auto t_network_end = std::chrono::high_resolution_clock::now();
    double network_ms = std::chrono::duration<double, std::milli>(t_network_end - t_network_start).count();
    
    if (res && res->status == 200) {
        auto result = nlohmann::json::parse(res->body);
        std::cout << "[Network] Response received in " << network_ms << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "Response:" << std::endl;
        std::cout << result["response"].get<std::string>() << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
    } else {
        std::cerr << "[Error] Request failed. Code: " << (res ? res->status : 0) << std::endl;
        return -1;
    }
    
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Final Summary:" << std::endl;
    std::cout << "  Model load: " << load_ms << " ms" << std::endl;
    std::cout << "  Edge encode: " << encode_ms << " ms" << std::endl;
    std::cout << "  Network RTT: " << network_ms << " ms" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    return 0;
}
