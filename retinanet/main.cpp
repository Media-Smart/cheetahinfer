#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/common.h"
#include "trtcommon/logger.h"

#include "timer.h"
#include "task/retinanet.h"

const std::string gSampleName = "TensorRT.retinanet";

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    //if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    //{
    //    params.dataDirs.push_back("data/mnist/");
    //    params.dataDirs.push_back("data/samples/mnist/");
    //}
    //else //!< Use the data directory provided by the user
    //{
    //    params.dataDirs = args.dataDirs;
    //}
    params.onnx_fp = "models/retinanet_1.onnx";
    params.engine_fp = "models/retinanet_1.engine";
    params.is_serialize = true;
	//params.inputTensorNames.push_back("Input3");
    //params.inputTensorNames.push_back("input.0");
    auto &tdims = params.input_dims;
    tdims.d[0] = 3;
    tdims.d[1] = 1088;
    tdims.d[2] = 1920;
    tdims.nbDims = 3;
    params.batch_size = 1;
	//params.outputTensorNames.push_back("Plus214_Output_0");
    //std::vector<std::string> names = {"scores", "boxes", "classes"};
    //{"output.0.0", "output.0.1", "output.1.0", "output.1.1", "output.2.0", "output.2.1", "output.3.0", "output.3.1", "output.4.0", "output.4.1"}; // .0 cls; .1 reg
    //"h3_reg", "h4_cls", "h4_reg", "h5_cls", "h5_reg", "h6_cls", "h6_reg", "h7_cls", "h7_reg"};
    //params.outputTensorNames.assign(names.begin(), names.end()); //push_back("resnetv15_dense0_fwd");
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

    cheetahinfer::RetinaNet retinanet(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for RetinaNet" << std::endl;

	cheetahinfer::Timer timer;
    retinanet.build(true);
	for(int i = 0; i < 1; i++)
	{
		timer.start();
		if (!retinanet.infer("data/resized.png"))
		{
			//return gLogger.reportFail(sampleTest);
		}

		timer.stop();
		//std::cout<<"infer#Elapse "<<timer.timeSpan()<<"s"<<std::endl;
        //std::cout<<"-------------"<<std::endl;
	}
    return gLogger.reportPass(sampleTest);
}

