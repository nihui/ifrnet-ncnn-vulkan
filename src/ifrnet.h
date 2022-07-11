// ifrnet implemented with ncnn library

#ifndef IFRNET_H
#define IFRNET_H

#include <string>

// ncnn
#include "net.h"

class IFRNet
{
public:
    IFRNet(int gpuid, bool tta_mode = false, int num_threads = 1);
    ~IFRNet();

#if _WIN32
    int load(const std::wstring& modeldir);
#else
    int load(const std::string& modeldir);
#endif

    int process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

    int process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net ifrnet;
    ncnn::Pipeline* ifrnet_preproc;
    ncnn::Pipeline* ifrnet_postproc;
    ncnn::Pipeline* ifrnet_timestep;
    bool tta_mode;
    int num_threads;
};

#endif // IFRNET_H
