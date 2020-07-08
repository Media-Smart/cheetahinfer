#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxConfig.h>

#include "utils/misc.h"
#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"
#include "task/base_task.h"

namespace cheetahinfer
{

    void BaseTask::build()
    {
        if (params_.is_from_onnx)
        {
            buildFromOnnx();
        }
        else
        {
            buildFromEngine();
        }

        check(nullptr != engine_, __FILE__, __LINE__, "Fail to build an engine");

        context_ = SampleUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        context_->setOptimizationProfile(0);
        context_->setBindingDimensions(0, params_.input_dims);

        getBindingDimensions();
        buffers_.init(engine_, context_);
    }

    void BaseTask::getBindingDimensions()
    {
        for (int ii = 1; ii < engine_->getNbBindings(); ii++)
        {
            auto output_dims = context_->getBindingDimensions(ii);
            output_dims_vec_.push_back(output_dims);
        }
    }

    void BaseTask::buildFromOnnx()
    {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        cheetahinfer::check(nullptr != builder, __FILE__, __LINE__, "Fail to create an infer builder");

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        check(nullptr != network, __FILE__, __LINE__, "Fail to create an network");

        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        check(nullptr != config, __FILE__, __LINE__, "Fail to create a configure");

        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        check(nullptr != parser, __FILE__, __LINE__, "Fail to create a parser");

        auto constructed = constructNetwork(builder, network, config, parser);
        check(true == constructed, __FILE__, __LINE__, "Fail to construct a network");

        addPlugin(network);

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        auto input_name = network->getInput(0)->getName();
        checkOptProfileDims(network->getInput(0)->getDimensions(), params_.input_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, params_.input_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, params_.input_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, params_.input_dims);

        config->addOptimizationProfile(profile);

        engine_ = SampleUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

        if (params_.is_serialize)
        {
            serializeEngine();
        }
    }

    void BaseTask::checkOptProfileDims(const nvinfer1::Dims& network_dims, const nvinfer1::Dims& setting_dims)
    {
        check(network_dims.nbDims == setting_dims.nbDims, __FILE__, __LINE__, "Number of dims not consistent");
        for (int ii = 0; ii < network_dims.nbDims; ii++)
        {
            check(setting_dims.d[ii] > 0, __FILE__, __LINE__, "There is dim less then 0 in setting dims");
            if (network_dims.d[ii] > 0)
            {
                check(network_dims.d[ii] == setting_dims.d[ii], __FILE__, __LINE__, "Dim of network dims and setting dims not consistent");
            }
        }

    }

    void BaseTask::addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
    {
    }

    void BaseTask::serializeEngine()
    {
        auto serializedModel = SampleUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
        sample::gLogInfo << "Serialize to file " << params_.engine_fp << std::endl;
        std::ofstream ofd(params_.engine_fp, std::ios::out | std::ios::binary);
        ofd.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
        //ofd.close(); // close file handler will result in a bug
    }

    void BaseTask::buildFromEngine()
    {
        check(isFileExists(params_.engine_fp), __FILE__, __LINE__, "Engine file not exists");
        std::ifstream ifd(params_.engine_fp, std::ios::in | std::ios::binary);
        ifd.seekg (0, ifd.end);
        size_t size = ifd.tellg();
        ifd.seekg (0, ifd.beg);

        std::unique_ptr<char> buffer(new char[size]);
        ifd.read(buffer.get(), size);
        ifd.close();

        auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

        engine_ = SampleUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.get(), size, nullptr));
    }

    bool BaseTask::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
            SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
            SampleUniquePtr<nvonnxparser::IParser>& parser)
    {
        check(isFileExists(params_.onnx_fp), __FILE__, __LINE__, "ONNX file not exists");
        auto parsed = parser->parseFromFile(
                params_.onnx_fp.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
        if (!parsed)
        {
            sample::gLogWarning << "Fail to parse the ONNX file " << params_.onnx_fp << std::endl; 
            return false;
        }

        config->setMaxWorkspaceSize(params_.max_workspace_size);
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

    bool BaseTask::infer(const std::vector<std::string>& fps)
    {
        return true;
    }

    bool BaseTask::inferCommon(const std::vector<std::string>& img_fps, samplesCommon::BufferManager& buffers)
    {
        // Read the input data into the managed buffers
        Timer timer;
        timer.start("inferCommon-processInput");
        processInput(img_fps, buffers);
        timer.stop("inferCommon-processInput");

        // Memcpy from host input buffers to device input buffers
        timer.start("inferCommon-H2D");
        buffers.copyInputToDevice();
        cudaDeviceSynchronize();
        timer.stop("inferCommon-H2D");

        timer.start("inferCommon-enqueue");
        bool status = context_->enqueueV2(buffers.getDeviceBindings().data(), _stream, nullptr);
        cudaStreamSynchronize(_stream);
        //cudaDeviceSynchronize();
        timer.stop("inferCommon-enqueue");

        if (!status)
        {
            sample::gLogWarning << "Fail to forward in TensorRT engine" << std::endl;  
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        timer.start("inferCommon-D2H");
        buffers.copyOutputToHost();
        cudaDeviceSynchronize();
        timer.start("inferCommon-D2H");

        return true;
    }

    void BaseTask::processInput(const std::vector<std::string>& img_fps, const samplesCommon::BufferManager& buffers)
    {
        float* host_data_buffer = static_cast<float*>(buffers.getHostBufferByIndex(0));
        readImages(img_fps, host_data_buffer);
    }

    void BaseTask::readImages(const std::vector<std::string>& fps, float* host_data)
    {
        auto& dims = params_.input_dims;
        const int stride = dims.d[1] * dims.d[2] * dims.d[3]; // c * h * w
        float* tmp_host_data = host_data;
        for (auto& fp : fps)
        {
            readImage(fp, tmp_host_data);
            tmp_host_data += stride;
        }
    }

    void BaseTask::readImage(const std::string& fp, float* host_data)
    {
        const int input_h = params_.input_dims.d[2];
        const int input_w = params_.input_dims.d[3];
        Timer timer;
        timer.start("readImage-imread");
        check(isFileExists(fp), __FILE__, __LINE__, "Image file not exists");
        orig_image_ = cv::imread(fp.c_str(), cv::IMREAD_COLOR);
        timer.stop("readImage-imread");
        if (orig_image_.empty())
        {
            check(true != orig_image_.empty(), __FILE__, __LINE__, "Fail to read an image");
        }
        cv::Mat image; // for debugging
        timer.start("readImage-resize");
        fx_ = double(input_w) / double(orig_image_.cols);
        fy_ = double(input_h) / double(orig_image_.rows);
        cv::resize(orig_image_, image, cv::Size(input_w, input_h));
        //cv::resize(orig_image_, image, cv::Size(), fx_, fy_);
        timer.stop("readImage-resize");

        int channels = 3;

        check(true == image.isContinuous(), __FILE__, __LINE__, "Image not continous");

        std::vector<float> mean {123.675, 116.28, 103.53}; //{0.485, 0.456, 0.406};
        std::vector<float> std {58.395, 57.12, 57.375}; //{0.229, 0.224, 0.225};

        unsigned char* img_data = (unsigned char*)image.data;
        timer.start("readImage-norm");
        for (int c = 0; c < channels; c++) {
            for (int j = 0, hw = input_w * input_h; j < hw; j++) {
                host_data[c * hw + j] = (float(img_data[channels * j + 2 - c]) - mean[c]) / std[c];
            }
        }        
        timer.stop("readImage-norm");
    }

    bool BaseTask::verifyOutput()
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

