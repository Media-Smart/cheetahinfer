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

#include "utils/misc.h"
#include "task/retinanet.h"

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    std::string savedir =  "models";
    std::string model_name = "retinanet_1"; //-v1-7";
    samplesCommon::OnnxSampleParams params;
    params.onnx_fp = savedir + "/" + model_name + ".onnx";
    params.engine_fp = savedir + "/" + model_name + ".engine";
    params.is_serialize = true;
    params.is_from_onnx = false;
    params.max_workspace_size = 128_MiB; // defined in trtcommon/common.h
    params.batch_size = 2;
    params.dla_core = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    auto &tdims = params.input_dims;
    tdims.d[0] = params.batch_size;
    tdims.d[1] = 3;
    tdims.d[2] = 1088;
    tdims.d[3] = 1920;
    tdims.nbDims = 4;

    return params;
}

void printHelpInfo()
{
    std::cout << "Usage: ./build/main [-h or --help] [--imgfp=<path to image>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--imgfp       Specify path to a image file path" << std::endl;
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
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    cheetahinfer::RetinaNet task(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for RetinaNet" << std::endl;

    cheetahinfer::Timer timer;
    task.build();
    for (int i = 0; i < 1; i++)
    {
        sample::gLogWarning << "---------------------------------------" <<std::endl;
        timer.start("main-infer");
        std::vector<std::string> fps = {args.imgfp, args.imgfp};
        if (!task.infer(fps))
        {
        }
        else
        {
            task.verifyOutput();
        }
        timer.stop("main-infer");
    }
    return 0;
}

