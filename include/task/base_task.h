//#pragma once
#ifndef CHEETAH_INFER_TASK_BASE_TASK_H
#define CHEETAH_INFER_TASK_BASE_TASK_H

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
    void build();
    bool inferCommon(const std::string fp, samplesCommon::BufferManager& buffers);
    virtual bool infer(std::string img_fp);
    virtual bool verifyOutput();

protected:
    // variables
    samplesCommon::OnnxSampleParams params_;
    nvinfer1::Dims input_dims_;
    std::vector<nvinfer1::Dims> output_dims_vec_;
    cudaStream_t _stream = nullptr;
    // do not modify as SampleUniquePtr<nvinfer1::IExecutionContext> engine_,
    // it will lead to compiling error which is caused by BufferManager defined
    // in trtcommon/buffers.h
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    SampleUniquePtr<nvinfer1::IExecutionContext> context_;
    samplesCommon::BufferManager buffers_;
    cv::Mat orig_image_; // for debugging
    float fx_;
    float fy_;

    // functions
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);
	void readImage(const std::string fp, float* host_data);
    void processInput(const std::string fp, const samplesCommon::BufferManager& buffers);
    void getBindingDimensions();
    void buildFromOnnx();
    void buildFromEngine();
    void serializeEngine();
    virtual void addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network);

};

} // namespace cheetahinfer

#endif
