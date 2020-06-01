#ifndef CHEETAH_INFER_BASE_TASK_H
#define CHEETAH_INFER_BASE_TASK_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>

#include "trtcommon/buffers.h"

namespace cheetahinfer{

class BaseTask
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    BaseTask(const samplesCommon::OnnxSampleParams& params);

    void build(bool is_from_onnx);

    bool inferCommon(const std::string fp, samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);

protected:
    samplesCommon::OnnxSampleParams params_; //!< The parameters for the sample.

    nvinfer1::Dims input_dims_;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> output_dims_vec_; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify
    cudaStream_t _stream = nullptr;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_; //!< The TensorRT engine used to run the network
    //SampleUniquePtr<nvinfer1::ICudaEngine> engine_;

    cv::Mat orig_image_;

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

	bool readImage(const std::string fp, float* host_data);

    bool processInput(const std::string fp, const samplesCommon::BufferManager& buffers);

    void getBindingDimensions();

    void buildFromOnnx();

    void buildFromPlan();

    void addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network);

    void serializeEngine();
};

}
#endif
