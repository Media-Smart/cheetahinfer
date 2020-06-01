#include <iostream>
#include <cstdlib>
#include <fstream>
//#include <sstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <cuda_runtime_api.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"

#include "timer.h"
#include "task/retinanet.h"
#include "plugin/decode_plugin.h"
#include "plugin/nms_plugin.h"

namespace cheetahinfer{

bool RetinaNet::build(bool isFromOnnx)
{
    if(isFromOnnx)
    {
        buildFromOnnx();
    }
    else
    {
        buildFromPlan();
    }

    //for(int ii=0; ii<mEngine->getNbBindings(); ii++)
    //{
    //    auto mOutputDims = mEngine->getBindingDimensions(ii);
    //    std::cout<<" Engine:mOutputDims "<<ii<<" "<<mOutputDims.nbDims<<" "<<mOutputDims.d[0]<<" "<<mOutputDims.d[1]<<" "<<mOutputDims.d[2]<<std::endl;
    //}

	if (!mEngine)
	{
		return false;
	}

	return true;
}

bool RetinaNet::buildFromOnnx()
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

    network->getInput(0)->setDimensions(mParams.input_dims);

	assert(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
    //std::cout<<"tdims"<<tdims.d[0]<<" "<<tdims.d[1]<<" "<<tdims.d[2]<<std::endl;
    //std::cout<<"mInputDims"<<mInputDims.d[0]<<" "<<mInputDims.d[1]<<" "<<mInputDims.d[2]<<std::endl;
	assert(mInputDims.nbDims == 3);

	assert(network->getNbOutputs() == 10);
    /*
    for(int i=0; i<10; i++)
    {
        nvinfer1::Dims mOutputDims = network->getOutput(i)->getDimensions();
        std::cout<<i<<" mOutputDims"<<" "<<mOutputDims.nbDims<<" "<<mOutputDims.d[0]<<" "<<mOutputDims.d[1]<<" "<<mOutputDims.d[2]<<std::endl;
        assert(mOutputDims.nbDims == 3);
        mOutputDimsVec.push_back(mOutputDims);
    }
    */

    // PULUGIN STARTS
    // Add decode plugins
    //std::cout << "Building accelerated plugins..." << std::endl;
    std::vector<nvinfer1::ITensor *> scores, boxes, classes;
    int nbOutputs = network->getNbOutputs();
    //int nbOutputs = mParams.outputTensorNames.size();
    //assert(nbOutputs == netNbOutputs);
    for (int i = 0; i < nbOutputs / 2; i++) {
        auto classOutput = network->getOutput(2 * i);
        std::cout << classOutput->getName() << std::endl;
        auto boxOutput = network->getOutput(2 * i + 1);
        auto outputDims = classOutput->getDimensions();

        int scale = mInputDims.d[2] / outputDims.d[2];
        //cout << "scale " << scale << " anchors " << endl;

        //auto a = anchors_[i];
        //for (int i = 0; i < a.size(); i++)
        //{
        //    cout << a[i] << endl;
        //}
        auto decodePlugin = DecodePlugin(score_thresh_, top_n_, anchors_[i], scale);
        std::vector<nvinfer1::ITensor *> inputs = {classOutput, boxOutput};
        auto layer = network->addPluginV2(inputs.data(), inputs.size(), decodePlugin);
        scores.push_back(layer->getOutput(0));
        boxes.push_back(layer->getOutput(1));
        classes.push_back(layer->getOutput(2));
    }

    // Cleanup outputs
    for (int i = 0; i < nbOutputs; i++) {
        auto output = network->getOutput(0);
        network->unmarkOutput(*output);
    }

    // Concat tensors from each feature map
    std::vector<nvinfer1::ITensor *> concat;
    for (auto tensors : {scores, boxes, classes}) {
        auto layer = network->addConcatenation(tensors.data(), tensors.size());
        concat.push_back(layer->getOutput(0));
    }
    
    // Add NMS plugin
    auto nmsPlugin = NMSPlugin(nms_thresh_, detections_per_im_);
    auto layer = network->addPluginV2(concat.data(), concat.size(), nmsPlugin); 
    //std::vector<std::string> &names = mParams.outputTensorNames; //{"scores", "boxes", "classes"};
    for (int i = 0; i < layer->getNbOutputs(); i++) {
        auto output = layer->getOutput(i);
        nvinfer1::Dims mOutputDims = output->getDimensions();
        //std::cout<<i<<" mOutputDims"<<" "<<mOutputDims.nbDims<<" "<<mOutputDims.d[0]<<" "<<mOutputDims.d[1]<<" "<<mOutputDims.d[2]<<std::endl;
        network->markOutput(*output);
        mOutputDimsVec.push_back(mOutputDims);
        //output->setName(names[i].c_str());
    }

    // Build engine
    //std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    //_engine = builder->buildEngineWithConfig(*network, *builderConfig);
    // PLUGIN ENDS

	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    mEngine = SampleUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

    auto serializedModel = SampleUniquePtr<nvinfer1::IHostMemory>(mEngine->serialize());
	//nvinfer1::IHostMemory *serializedModel = mEngine->serialize();
	std::ofstream file(mParams.engine_fp, std::ios::out | std::ios::binary);
	file.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
	//serializedModel->destroy();

    return true;
}

bool RetinaNet::buildFromPlan()
{
	std::ifstream ifile(mParams.engine_fp, std::ios::in | std::ios::binary);
	ifile.seekg (0, ifile.end);
	size_t size = ifile.tellg();
	ifile.seekg (0, ifile.beg);

	//char *buffer = new char[size];
    std::unique_ptr<char> buffer(new char[size]);
	ifile.read(buffer.get(), size);
	ifile.close();

	//nvinfer1::IRuntime *_runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
	auto _runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));

	//auto tEngine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
    mEngine = SampleUniquePtr<nvinfer1::ICudaEngine>(_runtime->deserializeCudaEngine(buffer.get(), size, nullptr));
	//delete[] buffer;
	//_runtime->destroy();

	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(tEngine, samplesCommon::InferDeleter());

    mInputDims = mEngine->getBindingDimensions(0);
    for(int ii = 1; ii < mEngine->getNbBindings(); ii++)
    {
        auto mOutputDims = mEngine->getBindingDimensions(ii);
        //std::cout<<" Engine:mOutputDims "<<ii<<" "<<mOutputDims.nbDims<<" "<<mOutputDims.d[0]<<" "<<mOutputDims.d[1]<<" "<<mOutputDims.d[2]<<std::endl;
        mOutputDimsVec.push_back(mOutputDims);
    }

    return true;
}

bool RetinaNet::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
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
    config->setMaxWorkspaceSize(1ULL << 30);
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


bool RetinaNet::infer()
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
    //assert(mParams.inputTensorNames.size() == 1);
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


bool RetinaNet::processInput(const samplesCommon::BufferManager& buffers)
{
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBufferByIndex(0)); //Name(mParams.inputTensorNames[0]));
	readImage("data/resized.png", hostDataBuffer);
    return true;
}


bool RetinaNet::readImage(const std::string fp, float* host_data)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
	std::cout << "inputH " << inputH << ", inputW " << inputW<<std::endl;
	std::cout << "Preparing data..." << std::endl;
	//auto image = cv::imread(fp.c_str(), cv::IMREAD_COLOR);
    orig_image_ = cv::imread(fp.c_str(), cv::IMREAD_COLOR);
    if (orig_image_.empty())
    {
        std::cout << "Image read failure" << std::endl;
    }
    std::cout << "Read image successfully" << std::endl;
	cv::resize(orig_image_, orig_image_, cv::Size(inputW, inputH));
	cv::Mat pixels;
	orig_image_.convertTo(pixels, CV_32FC3, 1.0, 0); // / 255

	int channels = 3;
	std::vector<float> img;

	if (pixels.isContinuous())
		img.assign((float*)pixels.datastart, (float*)pixels.dataend);
	else {
		std::cerr << "Error reading image " << fp << std::endl;
		return false;
	}

	std::vector<float> mean {123.675, 116.28, 103.53}; //{0.485, 0.456, 0.406};
	std::vector<float> std {58.395, 57.12, 57.375}; //{0.229, 0.224, 0.225};

	for (int c = 0; c < channels; c++) {
		for (int j = 0, hw = inputW * inputH; j < hw; j++) {
			host_data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
		}
	}        
	//for (int ii = 0; ii < channels * inputW * inputH; ii++) {
    //    cout << host_data[ii] << endl;
	//}        
	return true;
}


bool RetinaNet::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDimsVec[0].d[0];
    float* scores = static_cast<float*>(buffers.getHostBufferByIndex(1)); //Name("scores")); //mParams.outputTensorNames[0]));
    float* boxes = static_cast<float*>(buffers.getHostBufferByIndex(2)); //Name("boxes")); //mParams.outputTensorNames[0]));
    float* classes = static_cast<float*>(buffers.getHostBufferByIndex(3)); //Name("classes")); //mParams.outputTensorNames[0]));

    // Calculate Softmax
    for (int i = 0; i < outputSize; i++)
    {
        scores[i] = 1 / ( 1 + exp(-scores[i]));
    }

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        if (scores[i] > 0.9 && 0 == int(classes[i]))
        {
            int x1 = boxes[i*4+0];
            int y1 = boxes[i*4+1];
            int x2 = boxes[i*4+2];
            int y2 = boxes[i*4+3];
            std::cout << "Found box {" << x1 << ", " << y1 << ", " << x2 << ", " << y2
                << "} with score " << scores[i] << " and class " << classes[i] << std::endl;
            cv::rectangle(orig_image_, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
        }
    }
    std::string out_file = "det.jpg";
    cv::imwrite(out_file, orig_image_);
    gLogInfo << std::endl;

    return true;
}


/*
bool RetinaNet::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    char buffer[50];

    for(int ii = 0; ii < 5; ii ++)
    {
        // cls
        //sprintf(buffer, "output.%d.0");
        unsigned long long outputSize = mOutputDimsVec[2 * ii].d[0] * mOutputDimsVec[2 * ii].d[1] * mOutputDimsVec[2 * ii].d[2];
        float* cls = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2 * ii]));
        float* reg = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2 * ii + 1]));
        // reg
        //sprintf(buffer, "output.%d.1");

        cout << "====stride " << pow(2, ii + 3) << "====" << endl;
        cout << "####cls####" << endl;
        for (unsigned long long ii = 0; ii < outputSize; ii++) 
        {
            //cout << cls[ii] << endl;
            cout << ii << "/" << outputSize << ":" << 1 / ( 1 + exp(-cls[ii])) << endl;
        }        
        cout << "####reg####" << endl;
        for (unsigned long long ii = 0; ii < outputSize; ii++) 
        {
            cout << ii << "/" << outputSize << ":" << reg[ii] << endl;
        }        
    }

    gLogInfo << std::endl;

    return true;
}
*/



void RetinaNet::_prepare() 
{
    cudaStreamCreate(&_stream);
}


void RetinaNet::_set_hp()
{

	top_n_ = 1000;
    detections_per_im_ = 100;
	score_thresh_ = 0.9f;
	nms_thresh_ = 0.5;
    /*
    anchors_ = {
        {-12.0, -12.0, 20.0, 20.0, -7.31, -18.63, 15.31, 26.63, -18.63, -7.31, 26.63, 15.31, -16.16, -16.16, 24.16, 24.16, -10.25, -24.51, 18.25, 32.51, -24.51, -10.25, 32.51, 18.25, -21.4, -21.4, 29.4, 29.4, -13.96, -31.92, 21.96, 39.92, -31.92, -13.96, 39.92, 21.96},
        {-24.0, -24.0, 40.0, 40.0, -14.63, -37.25, 30.63, 53.25, -37.25, -14.63, 53.25, 30.63, -32.32, -32.32, 48.32, 48.32, -20.51, -49.02, 36.51, 65.02, -49.02, -20.51, 65.02, 36.51, -42.8, -42.8, 58.8, 58.8, -27.92, -63.84, 43.92, 79.84, -63.84, -27.92, 79.84, 43.92},
        {-48.0, -48.0, 80.0, 80.0, -29.25, -74.51, 61.25, 106.51, -74.51, -29.25, 106.51, 61.25, -64.63, -64.63, 96.63, 96.63, -41.02, -98.04, 73.02, 130.04, -98.04, -41.02, 130.04, 73.02, -85.59, -85.59, 117.59, 117.59, -55.84, -127.68, 87.84, 159.68, -127.68, -55.84, 159.68, 87.84},
        {-96.0, -96.0, 160.0, 160.0, -58.51, -149.02, 122.51, 213.02, -149.02, -58.51, 213.02, 122.51, -129.27, -129.27, 193.27, 193.27, -82.04, -196.07, 146.04, 260.07, -196.07, -82.04, 260.07, 146.04, -171.19, -171.19, 235.19, 235.19, -111.68, -255.35, 175.68, 319.35, -255.35, -111.68, 319.35, 175.68},
        {-192.0, -192.0, 320.0, 320.0, -117.02, -298.04, 245.02, 426.04, -298.04, -117.02, 426.04, 245.02, -258.54, -258.54, 386.54, 386.54, -164.07, -392.14, 292.07, 520.14, -392.14, -164.07, 520.14, 292.07, -342.37, -342.37, 470.37, 470.37, -223.35, -510.7, 351.35, 638.7, -510.7, -223.35, 638.7, 351.35}
    };
    */
    anchors_ = {
		{

        -19.,  -7.,  26.,  14.,
        -25., -10.,  32.,  17.,
        -32., -14.,  39.,  21.,

        -12., -12.,  19.,  19.,
        -16., -16.,  23.,  23.,
        -21., -21.,  28.,  28.,

        -7., -19.,  14.,  26.,
        -10., -25.,  17.,  32.,
        -14., -32.,  21.,  39.,
        },

 		{

        -37., -15.,  52.,  30.,
        -49., -21.,  64.,  36.,
        -64., -28.,  79.,  43.,

        -24., -24.,  39.,  39,
        -32., -32.,  47.,  47.,
        -43., -43.,  58.,  58.,

        -15., -37.,  30.,  52.,
        -21., -49.,  36.,  64.,
        -28., -64.,  43.,  79.,
        },

 		{

        -75.,  -29.,  106.,   60.,
        -98.,  -41.,  129.,   72.,
        -128.,  -56.,  159.,   87.,

        -48.,  -48.,   79.,   79.,
        -65.,  -65.,   96.,   96.,
        -86.,  -86.,  117.,  117.,

        -29.,  -75.,   60.,  106.,
        -41.,  -98.,   72.,  129.,
        -56., -128.,   87.,  159.,
        },

 		{

        -149.,  -59.,  212.,  122.,
        -196.,  -82.,  259.,  145.,
        -255., -112.,  318.,  175.,

        -96.,  -96.,  159.,  159.,
        -129., -129.,  192.,  192.,
        -171., -171.,  234.,  234.,

        -59., -149.,  122.,  212.,
        -82., -196.,  145.,  259.,
        -112., -255.,  175.,  318.,
        },

 		{

        -298., -117.,  425.,  244.,
        -392., -164.,  519.,  291.,
        -511., -223.,  638.,  350.,

        -192., -192.,  319.,  319.,
        -259., -259.,  386.,  386.,
        -342., -342.,  469.,  469.,

        -117., -298.,  244.,  425.,
        -164., -392.,  291.,  519.,
        -223., -511.,  350.,  638.,
        }
    };
}

}

/*

    anchors_ = {
		{

        -19.,  -7.,  26.,  14.,
        -25., -10.,  32.,  17.,
        -32., -14.,  39.,  21.,

        -12., -12.,  19.,  19.,
        -16., -16.,  23.,  23.,
        -21., -21.,  28.,  28.,

        -7., -19.,  14.,  26.,
        -10., -25.,  17.,  32.,
        -14., -32.,  21.,  39.,
        },

 		{

        -37., -15.,  52.,  30.,
        -49., -21.,  64.,  36.,
        -64., -28.,  79.,  43.,

        -24., -24.,  39.,  39,
        -32., -32.,  47.,  47.,
        -43., -43.,  58.,  58.,

        -15., -37.,  30.,  52.,
        -21., -49.,  36.,  64.,
        -28., -64.,  43.,  79.,
        },

 		{

        -75.,  -29.,  106.,   60.,
        -98.,  -41.,  129.,   72.,
        -128.,  -56.,  159.,   87.,

        -48.,  -48.,   79.,   79.,
        -65.,  -65.,   96.,   96.,
        -86.,  -86.,  117.,  117.,

        -29.,  -75.,   60.,  106.,
        -41.,  -98.,   72.,  129.,
        -56., -128.,   87.,  159.,
        },

 		{

        -149.,  -59.,  212.,  122.,
        -196.,  -82.,  259.,  145.,
        -255., -112.,  318.,  175.,

        -96.,  -96.,  159.,  159.,
        -129., -129.,  192.,  192.,
        -171., -171.,  234.,  234.,

        -59., -149.,  122.,  212.,
        -82., -196.,  145.,  259.,
        -112., -255.,  175.,  318.,
        },

 		{

        -298., -117.,  425.,  244.,
        -392., -164.,  519.,  291.,
        -511., -223.,  638.,  350.,

        -192., -192.,  319.,  319.,
        -259., -259.,  386.,  386.,
        -342., -342.,  469.,  469.,

        -117., -298.,  244.,  425.,
        -164., -392.,  291.,  519.,
        -223., -511.,  350.,  638.,
        }
    };
*/
