#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <cuda_runtime_api.h>

#include "trtcommon/argsParser.h"
#include "trtcommon/buffers.h"
#include "trtcommon/logger.h"

#include "utils/misc.h"
#include "task/retinanet.h"
#include "task/base_task.h"
#include "plugin/decode_plugin.h"
#include "plugin/nms_plugin.h"

namespace cheetahinfer{

void RetinaNet::addPlugin(SampleUniquePtr<nvinfer1::INetworkDefinition> &network)
{
	auto input_dims = network->getInput(0)->getDimensions();
    std::vector<nvinfer1::ITensor*> scores, boxes, classes;
    int nb_outputs = network->getNbOutputs();
    for (int i = 0; i < nb_outputs / 2; i++) {
        auto class_output = network->getOutput(2 * i);
        //gLogInfo << class_output->getName() << std::endl;
        auto box_output = network->getOutput(2 * i + 1);
        auto outputDims = class_output->getDimensions();

        int scale = input_dims.d[2] / outputDims.d[2];
        auto decode_plugin = DecodePlugin(score_thresh_, top_n_, anchors_[i], scale);
        std::vector<nvinfer1::ITensor*> inputs = {class_output, box_output};
        auto layer = network->addPluginV2(inputs.data(), inputs.size(), decode_plugin);
        scores.push_back(layer->getOutput(0));
        boxes.push_back(layer->getOutput(1));
        classes.push_back(layer->getOutput(2));
    }

    // Cleanup outputs
    for (int i = 0; i < nb_outputs; i++) {
        auto output = network->getOutput(0);
        network->unmarkOutput(*output);
    }

    // Concat tensors from each feature map
    std::vector<nvinfer1::ITensor*> concat;
    for (auto tensors : {scores, boxes, classes}) {
        auto layer = network->addConcatenation(tensors.data(), tensors.size());
        concat.push_back(layer->getOutput(0));
    }
    
    // Add NMS plugin
    auto nmsPlugin = NMSPlugin(nms_thresh_, detections_per_im_);
    auto layer = network->addPluginV2(concat.data(), concat.size(), nmsPlugin); 
    for (int i = 0; i < layer->getNbOutputs(); i++) {
        auto output = layer->getOutput(i);
        network->markOutput(*output);
    }
}

bool RetinaNet::verifyOutput()
{
    const int output_size = output_dims_vec_[0].d[0];
    float* scores = static_cast<float*>(buffers_.getHostBufferByIndex(1));
    float* boxes = static_cast<float*>(buffers_.getHostBufferByIndex(2));
    float* classes = static_cast<float*>(buffers_.getHostBufferByIndex(3));

    // Calculate Softmax
    for (int i = 0; i < output_size; i++)
    {
        scores[i] = 1 / ( 1 + exp(-scores[i]));
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < output_size; i++)
    {
        if (scores[i] > 0.9 && 0 == int(classes[i]))
        {
            int x1 = float(boxes[i*4+0] / fx_);
            int y1 = float(boxes[i*4+1] / fy_);
            int x2 = float(boxes[i*4+2] / fx_);
            int y2 = float(boxes[i*4+3] / fy_);
            gLogInfo << "Found box {" << x1 << ", " << y1 << ", " << x2 << ", " << y2
                << "} with score " << scores[i] << " and class " << classes[i] << std::endl;
            cv::rectangle(orig_image_, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
        }
    }
    //std::string out_file = "det.jpg";
    //cv::imwrite(out_file, orig_image_);
    gLogInfo << std::endl;

    return true;
}

bool RetinaNet::infer(std::string img_fp)
{
    return inferCommon(img_fp, buffers_);
}

RetinaNet::RetinaNet(const samplesCommon::OnnxSampleParams& params)
    : BaseTask(params)
{
    setHp();
}

void RetinaNet::setHp()
{
	top_n_ = 1000;
    detections_per_im_ = 100;
	score_thresh_ = 0.9f;
	nms_thresh_ = 0.5;
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

} //namespace cheetahinfer

