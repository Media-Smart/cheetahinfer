#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxConfig.h>

#include "timer.h"
#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"
#include "plugin/decode_plugin.h"
#include "plugin/nms_plugin.h"
#include "task/base_task.h"

namespace cheetahinfer
{


void BaseTask::build(bool is_from_onnx)
{
    if(is_from_onnx)
    {
        buildFromOnnx();
    }
    else
    {
        buildFromPlan();
    }

    getBindingDimensions();

    assert(engine_);
}

void BaseTask::getBindingDimensions()
{
    input_dims_ = engine_->getBindingDimensions(0);
    for(int ii = 1; ii < engine_->getNbBindings(); ii++)
    {
        auto output_dims = engine_->getBindingDimensions(ii);
        //std::cout<<ii<<" mOutputDims"<<" "<<output_dims.nbDims<<" "<<output_dims.d[0]<<" "<<output_dims.d[1]<<" "<<output_dims.d[2]<<std::endl;
        output_dims_vec_.push_back(output_dims);
    }
}

void BaseTask::buildFromOnnx()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    assert(builder);

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    assert(network);

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    assert(config);

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    assert(parser);

    auto constructed = constructNetwork(builder, network, config, parser);
    assert(constructed);

    network->getInput(0)->setDimensions(params_.input_dims);

    addPlugin(network);

    engine_ = SampleUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    std::cout << params_.is_serialize << " " << params_.engine_fp << std::endl;
    if (params_.is_serialize)
    {
        serializeEngine();
    }
}

void BaseTask::addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
{
}

void BaseTask::serializeEngine()
{
    auto serializedModel = SampleUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
    std::cout << params_.engine_fp << std::endl;
    std::ofstream ofd(params_.engine_fp, std::ios::out | std::ios::binary);
    ofd.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    //ofd.close();
}

void BaseTask::buildFromPlan()
{
    std::ifstream ifd(params_.engine_fp, std::ios::in | std::ios::binary);
    ifd.seekg (0, ifd.end);
    size_t size = ifd.tellg();
    ifd.seekg (0, ifd.beg);

    std::unique_ptr<char> buffer(new char[size]);
    ifd.read(buffer.get(), size);
    ifd.close();

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));

    engine_ = SampleUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.get(), size, nullptr));
}

bool BaseTask::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        params_.onnx_fp.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(params_.batch_size);
    config->setMaxWorkspaceSize(1ULL << 30);
    if (params_.fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (params_.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), params_.dla_core);

    return true;
}

bool BaseTask::inferCommon(std::string img_fp, samplesCommon::BufferManager& buffers)
{
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    assert(context);
    //if (!context)
    //{
    //    return false;
    //}

    // Read the input data into the managed buffers
    assert(processInput(img_fp, buffers));
    //if (!processInput(img_fp, buffers))
    //{
    //    return false;
    //}

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    cudaDeviceSynchronize();

    bool status = context->enqueue(params_.batch_size, buffers.getDeviceBindings().data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
    //cudaDeviceSynchronize();

    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    cudaDeviceSynchronize();

    return true;
}

bool BaseTask::processInput(const std::string img_fp, const samplesCommon::BufferManager& buffers)
{
    float* host_data_buffer = static_cast<float*>(buffers.getHostBufferByIndex(0)); //Name(params_.inputTensorNames[0]));
    return readImage(img_fp, host_data_buffer);
}

bool BaseTask::readImage(const std::string fp, float* host_data)
{
    const int input_h = input_dims_.d[1];
    const int input_w = input_dims_.d[2];
    orig_image_ = cv::imread(fp.c_str(), cv::IMREAD_COLOR);
    if (orig_image_.empty())
    {
        std::cerr << "Error reading image " << fp << std::endl;
        return false;
    }
    cv::resize(orig_image_, orig_image_, cv::Size(input_w, input_h));
    cv::Mat pixels;
    orig_image_.convertTo(pixels, CV_32FC3, 1.0, 0); // / 255

    int channels = 3;
    std::vector<float> img;

    if (pixels.isContinuous())
    {
        img.assign((float*)pixels.datastart, (float*)pixels.dataend);
    }
    else
    {
        std::cerr << "Image not continous" << std::endl;
        return false;
    }


    std::vector<float> mean {123.675, 116.28, 103.53}; //{0.485, 0.456, 0.406};
    std::vector<float> std {58.395, 57.12, 57.375}; //{0.229, 0.224, 0.225};

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = input_w * input_h; j < hw; j++) {
            host_data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
        }
    }        
    return true;
}

bool BaseTask::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    return true;
}

BaseTask::BaseTask(const samplesCommon::OnnxSampleParams& params)
    : params_(params)
    , engine_(nullptr)
{
    cudaStreamCreate(&_stream);
}


} //namespace cheetahinfer

