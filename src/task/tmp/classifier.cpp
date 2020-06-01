//#include <cstdlib>
//#include <fstream>
#include <iostream>
//#include <sstream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"

#include "timer.h"
#include "task/classifier.h"


namespace cheetahinfer{

bool Classifier::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}

	/*
	std::ifstream ifile("models/resnet18-v1-7.engine", std::ios::in | std::ios::binary);
	ifile.seekg (0, ifile.end);
	size_t size = ifile.tellg();
	ifile.seekg (0, ifile.beg);

	char *buffer = new char[size];
	ifile.read(buffer, size);
	ifile.close();

	nvinfer1::IRuntime *_runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

	auto tEngine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
	delete[] buffer;
	_runtime->destroy()

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(tEngine, samplesCommon::InferDeleter());
	*/

    nvinfer1::Dims tdims;
    tdims.d[0] = 3;
    tdims.d[1] = 299;
    tdims.d[2] = 299;
    tdims.nbDims = 3;

    network->getInput(0)->setDimensions(tdims);

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
    std::cout<<"tdims"<<tdims.d[0]<<" "<<tdims.d[1]<<" "<<tdims.d[2]<<std::endl;
    std::cout<<"mInputDims"<<mInputDims.d[0]<<" "<<mInputDims.d[1]<<" "<<mInputDims.d[2]<<std::endl;
	assert(mInputDims.nbDims == 3);

	assert(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
    std::cout<<"mOutputDims"<<" "<<mOutputDims.nbDims<<" "<<mOutputDims.d[0]<<" "<<mOutputDims.d[1]<<" "<<mOutputDims.d[2]<<std::endl;
	assert(mOutputDims.nbDims == 1);

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

	//nvinfer1::IHostMemory *serializedModel = mEngine->serialize();
	//std::ofstream file("models/resnet18-v1-7.engine", std::ios::out | std::ios::binary);
	//file.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
	//serializedModel->destroy();

	if (!mEngine)
	{
		return false;
	}


	return true;
}


bool Classifier::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        mParams.onnx_fp.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batch_size);
    config->setMaxWorkspaceSize(1 << 30);
    if (mParams.fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dla_core);

    return true;
}


bool Classifier::infer()
{
    // Create RAII buffer manager object
	std::cout<<"xxxxxxx buffer def"<<std::endl;
    samplesCommon::BufferManager buffers(mEngine, mParams.batch_size);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.input_tensor_names.size() == 1);
	cheetahinfer::Timer timer;
	timer.start();
    if (!processInput(buffers))
    {
        return false;
    }
	timer.stop();
	std::cout<<"processInput#Elapse "<<timer.timeSpan()<<"s"<<std::endl;

    // Memcpy from host input buffers to device input buffers
	timer.start();
    buffers.copyInputToDevice();
	cudaDeviceSynchronize();
	timer.stop();
	std::cout<<"copyInputToDevice#Elapse "<<timer.timeSpan()<<"s"<<std::endl;

	timer.start();
    //bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
	//cudaDeviceSynchronize();

    bool status = context->enqueue(mParams.batch_size, buffers.getDeviceBindings().data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
	//cudaDeviceSynchronize();

	timer.stop();
	std::cout<<"execute#Elapse "<<timer.timeSpan()<<"s"<<std::endl;
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
	timer.start();
    buffers.copyOutputToHost();
	cudaDeviceSynchronize();
	timer.stop();
	std::cout<<"copyOutputToHost#Elapse "<<timer.timeSpan()<<"s"<<std::endl;

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}


bool Classifier::processInput(const samplesCommon::BufferManager& buffers)
{
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBufferByName(mParams.input_tensor_names[0]));
	readImage("data/cat.jpg", hostDataBuffer);
    return true;
}

bool Classifier::readImage(const std::string fp, float* host_data)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
	std::cout << "inputH " << inputH << ", inputW " << inputW<<std::endl;
	std::cout << "Preparing data..." << std::endl;
	auto image = cv::imread(fp.c_str(), cv::IMREAD_COLOR);
	cv::resize(image, image, cv::Size(inputW, inputH));
	cv::Mat pixels;
	image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

	int channels = 3;
	std::vector<float> img;

	if (pixels.isContinuous())
		img.assign((float*)pixels.datastart, (float*)pixels.dataend);
	else {
		std::cerr << "Error reading image " << fp << std::endl;
		return false;
	}

	std::vector<float> mean {0.485, 0.456, 0.406};
	std::vector<float> std {0.229, 0.224, 0.225};

	for (int c = 0; c < channels; c++) {
		for (int j = 0, hw = inputW * inputH; j < hw; j++) {
			host_data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
		}
	}        
	return true;
}


bool Classifier::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[0];
    float* output = static_cast<float*>(buffers.getHostBufferByName(mParams.output_tensor_names[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        std::cout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
                 //<< "Class " << i << ": " << std::endl;
    }
    gLogInfo << std::endl;

    return idx == mNumber && val > 0.9f;
}


void Classifier::_prepare() 
{
    cudaStreamCreate(&_stream);
}

}
