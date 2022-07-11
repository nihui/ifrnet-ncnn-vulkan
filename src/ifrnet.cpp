// ifrnet implemented with ncnn library

#include "ifrnet.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "ifrnet_preproc.comp.hex.h"
#include "ifrnet_preproc_tta.comp.hex.h"
#include "ifrnet_postproc.comp.hex.h"
#include "ifrnet_postproc_tta.comp.hex.h"
#include "ifrnet_timestep.comp.hex.h"
#include "ifrnet_timestep_tta.comp.hex.h"

#include "ifrnet_ops.h"

DEFINE_LAYER_CREATOR(Warp)

IFRNet::IFRNet(int gpuid, bool _tta_mode, int _num_threads)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    ifrnet_preproc = 0;
    ifrnet_postproc = 0;
    ifrnet_timestep = 0;
    tta_mode = _tta_mode;
    num_threads = _num_threads;
}

IFRNet::~IFRNet()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete ifrnet_preproc;
        delete ifrnet_postproc;
        delete ifrnet_timestep;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int IFRNet::load(const std::wstring& modeldir)
#else
int IFRNet::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;

    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = false;

    ifrnet.opt = opt;

    ifrnet.set_vulkan_device(vkdev);

    ifrnet.register_custom_layer("ifrnet.Warp", Warp_layer_creator);

#if _WIN32
    load_param_model(ifrnet, modeldir, L"ifrnet");
#else
    load_param_model(ifrnet, modeldir, "ifrnet");
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(ifrnet_preproc_tta_comp_data, sizeof(ifrnet_preproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(ifrnet_preproc_comp_data, sizeof(ifrnet_preproc_comp_data), opt, spirv);
                }
            }

            ifrnet_preproc = new ncnn::Pipeline(vkdev);
            ifrnet_preproc->set_optimal_local_size_xyz(8, 8, 3);
            ifrnet_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(ifrnet_postproc_tta_comp_data, sizeof(ifrnet_postproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(ifrnet_postproc_comp_data, sizeof(ifrnet_postproc_comp_data), opt, spirv);
                }
            }

            ifrnet_postproc = new ncnn::Pipeline(vkdev);
            ifrnet_postproc->set_optimal_local_size_xyz(8, 8, 3);
            ifrnet_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(ifrnet_timestep_tta_comp_data, sizeof(ifrnet_timestep_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(ifrnet_timestep_comp_data, sizeof(ifrnet_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            ifrnet_timestep = new ncnn::Pipeline(vkdev);
            ifrnet_timestep->set_optimal_local_size_xyz(8, 8, 1);
            ifrnet_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int IFRNet::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (!vkdev)
    {
        // cpu only
        return process_cpu(in0image, in1image, timestep, outimage);
    }

    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    fprintf(stderr, "timestep = %f\n", timestep);

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = ifrnet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        in0 = ncnn::Mat(w, h, (unsigned char*)pixel0data, (size_t)channels, 1);
        in1 = ncnn::Mat(w, h, (unsigned char*)pixel1data, (size_t)channels, 1);
    }
    else
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
    }

    ncnn::VkMat out_gpu;

    if (tta_mode)
    {
        // preproc
        ncnn::VkMat in0_gpu_padded[8];
        ncnn::VkMat in1_gpu_padded[8];
        ncnn::VkMat timestep_gpu_padded[2];
        {
            in0_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in0_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded[0];
            bindings[2] = in0_gpu_padded[1];
            bindings[3] = in0_gpu_padded[2];
            bindings[4] = in0_gpu_padded[3];
            bindings[5] = in0_gpu_padded[4];
            bindings[6] = in0_gpu_padded[5];
            bindings[7] = in0_gpu_padded[6];
            bindings[8] = in0_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded[0].w;
            constants[4].i = in0_gpu_padded[0].h;
            constants[5].i = in0_gpu_padded[0].cstep;

            cmd.record_pipeline(ifrnet_preproc, bindings, constants, in0_gpu_padded[0]);
        }
        {
            in1_gpu_padded[0].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[1].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[2].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[3].create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[4].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[5].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[6].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);
            in1_gpu_padded[7].create(h_padded, w_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded[0];
            bindings[2] = in1_gpu_padded[1];
            bindings[3] = in1_gpu_padded[2];
            bindings[4] = in1_gpu_padded[3];
            bindings[5] = in1_gpu_padded[4];
            bindings[6] = in1_gpu_padded[5];
            bindings[7] = in1_gpu_padded[6];
            bindings[8] = in1_gpu_padded[7];

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded[0].w;
            constants[4].i = in1_gpu_padded[0].h;
            constants[5].i = in1_gpu_padded[0].cstep;

            cmd.record_pipeline(ifrnet_preproc, bindings, constants, in1_gpu_padded[0]);
        }
        {
            timestep_gpu_padded[0].create(w_padded / 16, h_padded / 16, 1, in_out_tile_elemsize, 1, blob_vkallocator);
            timestep_gpu_padded[1].create(h_padded / 16, w_padded / 16, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = timestep_gpu_padded[0];
            bindings[1] = timestep_gpu_padded[1];

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded[0].w;
            constants[1].i = timestep_gpu_padded[0].h;
            constants[2].i = timestep_gpu_padded[0].cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(ifrnet_timestep, bindings, constants, timestep_gpu_padded[0]);
        }

        ncnn::VkMat out_gpu_padded[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // ifrnet
            ncnn::Extractor ex = ifrnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded[ti]);
            ex.input("in1", in1_gpu_padded[ti]);
            ex.input("in2", timestep_gpu_padded[ti / 4]);
            ex.extract("out0", out_gpu_padded[ti], cmd);
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(9);
            bindings[0] = out_gpu_padded[0];
            bindings[1] = out_gpu_padded[1];
            bindings[2] = out_gpu_padded[2];
            bindings[3] = out_gpu_padded[3];
            bindings[4] = out_gpu_padded[4];
            bindings[5] = out_gpu_padded[5];
            bindings[6] = out_gpu_padded[6];
            bindings[7] = out_gpu_padded[7];
            bindings[8] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded[0].w;
            constants[1].i = out_gpu_padded[0].h;
            constants[2].i = out_gpu_padded[0].cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(ifrnet_postproc, bindings, constants, out_gpu);
        }
    }
    else
    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        ncnn::VkMat timestep_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(ifrnet_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(ifrnet_preproc, bindings, constants, in1_gpu_padded);
        }
        {
            timestep_gpu_padded.create(w_padded / 16, h_padded / 16, 1, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(1);
            bindings[0] = timestep_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].i = timestep_gpu_padded.w;
            constants[1].i = timestep_gpu_padded.h;
            constants[2].i = timestep_gpu_padded.cstep;
            constants[3].f = timestep;

            cmd.record_pipeline(ifrnet_timestep, bindings, constants, timestep_gpu_padded);
        }

        // ifrnet
        ncnn::VkMat out_gpu_padded;
        {
            ncnn::Extractor ex = ifrnet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.input("in2", timestep_gpu_padded);
            ex.extract("out0", out_gpu_padded, cmd);
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(ifrnet_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        if (!(opt.use_fp16_storage && opt.use_int8_storage))
        {
#if _WIN32
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int IFRNet::process_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::Option opt = ifrnet.opt;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    ncnn::Mat in0;
    ncnn::Mat in1;
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::Mat out;

    if (tta_mode)
    {
        // preproc and border padding
        ncnn::Mat in0_padded[8];
        ncnn::Mat in1_padded[8];
        ncnn::Mat timestep_padded[2];
        {
            in0_padded[0].create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded[0].channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded[0].create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded[0].channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            timestep_padded[0].create(h_padded / 16, w_padded / 16, 1);
            timestep_padded[1].create(h_padded / 16, w_padded / 16, 1);
            timestep_padded[0].fill(timestep);
            timestep_padded[1].fill(timestep);
        }

        // the other 7 directions
        {
            in0_padded[1].create(w_padded, h_padded, 3);
            in0_padded[2].create(w_padded, h_padded, 3);
            in0_padded[3].create(w_padded, h_padded, 3);
            in0_padded[4].create(h_padded, w_padded, 3);
            in0_padded[5].create(h_padded, w_padded, 3);
            in0_padded[6].create(h_padded, w_padded, 3);
            in0_padded[7].create(h_padded, w_padded, 3);

            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat in0_padded_0 = in0_padded[0].channel(q);
                ncnn::Mat in0_padded_1 = in0_padded[1].channel(q);
                ncnn::Mat in0_padded_2 = in0_padded[2].channel(q);
                ncnn::Mat in0_padded_3 = in0_padded[3].channel(q);
                ncnn::Mat in0_padded_4 = in0_padded[4].channel(q);
                ncnn::Mat in0_padded_5 = in0_padded[5].channel(q);
                ncnn::Mat in0_padded_6 = in0_padded[6].channel(q);
                ncnn::Mat in0_padded_7 = in0_padded[7].channel(q);

                for (int i = 0; i < h_padded; i++)
                {
                    const float* outptr0 = in0_padded_0.row(i);
                    float* outptr1 = in0_padded_1.row(i) + w_padded - 1;
                    float* outptr2 = in0_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    float* outptr3 = in0_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w_padded; j++)
                    {
                        float* outptr4 = in0_padded_4.row(j) + i;
                        float* outptr5 = in0_padded_5.row(j) + h_padded - 1 - i;
                        float* outptr6 = in0_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        float* outptr7 = in0_padded_7.row(w_padded - 1 - j) + i;

                        float v = *outptr0++;

                        *outptr1-- = v;
                        *outptr2-- = v;
                        *outptr3++ = v;
                        *outptr4 = v;
                        *outptr5 = v;
                        *outptr6 = v;
                        *outptr7 = v;
                    }
                }
            }
        }
        {
            in1_padded[1].create(w_padded, h_padded, 3);
            in1_padded[2].create(w_padded, h_padded, 3);
            in1_padded[3].create(w_padded, h_padded, 3);
            in1_padded[4].create(h_padded, w_padded, 3);
            in1_padded[5].create(h_padded, w_padded, 3);
            in1_padded[6].create(h_padded, w_padded, 3);
            in1_padded[7].create(h_padded, w_padded, 3);

            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat in1_padded_0 = in1_padded[0].channel(q);
                ncnn::Mat in1_padded_1 = in1_padded[1].channel(q);
                ncnn::Mat in1_padded_2 = in1_padded[2].channel(q);
                ncnn::Mat in1_padded_3 = in1_padded[3].channel(q);
                ncnn::Mat in1_padded_4 = in1_padded[4].channel(q);
                ncnn::Mat in1_padded_5 = in1_padded[5].channel(q);
                ncnn::Mat in1_padded_6 = in1_padded[6].channel(q);
                ncnn::Mat in1_padded_7 = in1_padded[7].channel(q);

                for (int i = 0; i < h_padded; i++)
                {
                    const float* outptr0 = in1_padded_0.row(i);
                    float* outptr1 = in1_padded_1.row(i) + w_padded - 1;
                    float* outptr2 = in1_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    float* outptr3 = in1_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w_padded; j++)
                    {
                        float* outptr4 = in1_padded_4.row(j) + i;
                        float* outptr5 = in1_padded_5.row(j) + h_padded - 1 - i;
                        float* outptr6 = in1_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        float* outptr7 = in1_padded_7.row(w_padded - 1 - j) + i;

                        float v = *outptr0++;

                        *outptr1-- = v;
                        *outptr2-- = v;
                        *outptr3++ = v;
                        *outptr4 = v;
                        *outptr5 = v;
                        *outptr6 = v;
                        *outptr7 = v;
                    }
                }
            }
        }

        ncnn::Mat out_padded[8];
        for (int ti = 0; ti < 8; ti++)
        {
            // ifrnet
            {
                ncnn::Extractor ex = ifrnet.create_extractor();

                ex.input("in0", in0_padded[ti]);
                ex.input("in1", in1_padded[ti]);
                ex.input("in2", timestep_padded[ti / 4]);
                ex.extract("out0", out_padded[ti]);
            }
        }

        // cut padding and postproc
        out.create(w, h, 3);
        {
            for (int q = 0; q < 3; q++)
            {
                const ncnn::Mat out_padded_0 = out_padded[0].channel(q);
                const ncnn::Mat out_padded_1 = out_padded[1].channel(q);
                const ncnn::Mat out_padded_2 = out_padded[2].channel(q);
                const ncnn::Mat out_padded_3 = out_padded[3].channel(q);
                const ncnn::Mat out_padded_4 = out_padded[4].channel(q);
                const ncnn::Mat out_padded_5 = out_padded[5].channel(q);
                const ncnn::Mat out_padded_6 = out_padded[6].channel(q);
                const ncnn::Mat out_padded_7 = out_padded[7].channel(q);
                float* outptr = out.channel(q);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr0 = out_padded_0.row(i);
                    const float* ptr1 = out_padded_1.row(i) + w_padded - 1;
                    const float* ptr2 = out_padded_2.row(h_padded - 1 - i) + w_padded - 1;
                    const float* ptr3 = out_padded_3.row(h_padded - 1 - i);

                    for (int j = 0; j < w; j++)
                    {
                        const float* ptr4 = out_padded_4.row(j) + i;
                        const float* ptr5 = out_padded_5.row(j) + h_padded - 1 - i;
                        const float* ptr6 = out_padded_6.row(w_padded - 1 - j) + h_padded - 1 - i;
                        const float* ptr7 = out_padded_7.row(w_padded - 1 - j) + i;

                        float v = (*ptr0++ + *ptr1-- + *ptr2-- + *ptr3++ + *ptr4 + *ptr5 + *ptr6 + *ptr7) / 8;

                        *outptr++ = (v + 0.5f) * 255.f + 0.5f;
                    }
                }
            }
        }
    }
    else
    {
        // preproc and border padding
        ncnn::Mat in0_padded;
        ncnn::Mat in1_padded;
        ncnn::Mat timestep_padded;
        {
            in0_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f) - 0.5f;
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            timestep_padded.create(w_padded / 16, h_padded / 16, 1);
            timestep_padded.fill(timestep);
        }

        // ifrnet
        ncnn::Mat out_padded;
        {
            ncnn::Extractor ex = ifrnet.create_extractor();

            ex.input("in0", in0_padded);
            ex.input("in1", in1_padded);
            ex.input("in2", timestep_padded);
            ex.extract("out0", out_padded);
        }

        // cut padding and postproc
        out.create(w, h, 3);
        {
            for (int q = 0; q < 3; q++)
            {
                float* outptr = out.channel(q);
                const float* ptr = out_padded.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = (*ptr++ + 0.5f) * 255.f + 0.5f;
                    }
                }
            }
        }
    }

    // download
    {
#if _WIN32
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
    }

    return 0;
}
