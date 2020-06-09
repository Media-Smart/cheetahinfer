#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/common.h"
#include "trtcommon/logger.h"
//#include "parserOnnxConfig.h"

#include "timer.h"
#include "task/classifier.h"


const std::string gSampleName = "TensorRT.classifer";


samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    params.onnx_fp = "models/resnet18-v1-7.onnx";
	//params.inputTensorNames.push_back("Input3");
    params.input_tensor_names.push_back("data");
    params.batch_size = 16;
	//params.outputTensorNames.push_back("Plus214_Output_0");
    params.output_tensor_names.push_back("resnetv15_dense0_fwd");
    params.dla_core = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}


void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    cheetahinfer::Classifier classifer(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

	cheetahinfer::Timer timer;
    if (!classifer.build())
    {
        return gLogger.reportFail(sampleTest);
    }
	for(int i=0; i<1; i++)
	{
		timer.start();
		if (!classifer.infer())
		{
			//return gLogger.reportFail(sampleTest);
		}

		timer.stop();
		std::cout<<"infer#Elapse "<<timer.timeSpan()<<"s"<<std::endl;
		std::cout<<"-------------"<<std::endl;
	}
    return gLogger.reportPass(sampleTest);
}

