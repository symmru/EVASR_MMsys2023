#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/eval.h"
#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/parseutils.h"
#include "libavutil/log.h"
#include "libswscale/swscale.h"
#include "avfilter.h"
#include "internal.h"
#include "video.h"

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <sys/time.h>

#include "edsr/pytorch_model.h"

long per_frame_time = 0;
int i = 0;
char** patch_info;

typedef struct EDSRContext {
    const AVClass* class;
    AVDictionary* opts;
    uint8_t* data;
    int frame_width;
    int frame_height;
    int patch_width;
    int patch_height;
    int scale;
    char* file_path;
    char* model_path;
    struct CPytorchModel* model;
} EDSRContext;

static int readLine(FILE* fp, char* buf, int size)
{
    char ch;
    int count = 0;
    while(++count <= size && ((ch = getc(fp)) != '\n'))
    {
        if(ch == EOF)
        {
            count --;
            return count;
        }
        buf[count-1] = ch;
    }
    buf[count-1] = '\0';
    return count;
}

static int load_file(char* file_path)
{
    // av_log(NULL, AV_LOG_INFO, "--------load file called---------\n");
    int linesize = 8992; // frame+1
    patch_info = malloc(linesize*sizeof(char*));
    for(int i=0;i<linesize;i++)
    {
        patch_info[i] = malloc(64*sizeof(char));
    }
    FILE* fp;
    int ret;
    char line[64];
    fp = fopen(file_path,"r");
    if(fp == NULL)
        av_log(NULL, AV_LOG_INFO, "--------failed to open file---------\n");
    int i = 0;
    while(ret = readLine(fp, line, 64) > 0)
    {
        memcpy(patch_info[i], line, 64);
        i+=1;
    }
    fclose(fp);
    return 0;
}

//configure output based on transform options - update link
static int config_output(AVFilterLink* outlink) {
    AVFilterContext* ctx = outlink->src;
    EDSRContext* s = ctx->priv;

    // If you want to scale the output resolution, do it here.
    // Changing the output resolution in filter_frame changes it on each frame, so a scale of 2 becomes x4, x8, x16, etc.
    outlink->w = s->frame_width * s->scale;
    outlink->h = s->frame_height * s->scale;

    return 0;
}

// First callback - "Note that at this point, your local context already has the user options initialized,
// but you still haven't any clue about the kind of data input you will get"
static av_cold int init_dict(AVFilterContext *ctx, AVDictionary **opts) {
    EDSRContext* s = ctx->priv;
    // av_log(NULL, AV_LOG_INFO, "--------init model---------\n");
    s->model = CPytorchModel_init(s->model_path);
    s->opts = *opts;

    *opts = NULL;
    load_file(s->file_path);
    return 0;
}

static av_cold void uninit(AVFilterContext* filter_context) {
    EDSRContext* s = filter_context->priv;
    // av_log(NULL, AV_LOG_INFO, "--------uninit model---------\n");
    CPytorchModel_del(s->model);
    av_dict_free(&s->opts);
    s->opts = NULL;
    free(patch_info);
}



static int filter_frame(AVFilterLink* inlink, AVFrame* in) {

    struct timeval stv, etv;
    gettimeofday(&stv, NULL);

    AVFilterContext* context = inlink->dst; // Local context from input link
    EDSRContext* s = context->priv;
    AVFilterLink* outlink = context->outputs[0]; // pointer to context output pad

    // load_file(s->file_path);
    char* current_patch = malloc(64*sizeof(char));
    current_patch = patch_info[in->coded_picture_number];
    char c_num_batches = current_patch[0];
    int num_batches = c_num_batches;
    num_batches = num_batches - 65;

    // upscale y u v 
    struct SwsContext *scale;
    scale = sws_getContext(in->width,in->height,
        AV_PIX_FMT_GRAY8, outlink->w, outlink->h, AV_PIX_FMT_GRAY8, 4, NULL,NULL,NULL);

    uint8_t* input;
    int in_y_pixels = in->width * in->height;
    input = (uint8_t*) malloc(in_y_pixels);
    
    for(int j=0;j<in->height;j++)
    {
        memcpy(input+j*in->width, in->data[0]+j*in->linesize[0], sizeof(uint8_t)*in->width);
    }


    AVFrame* out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    // Throw an error if we're out of memory
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    // to output link
    av_frame_copy_props(out, in);
    int out_y_pixels = outlink->w * outlink->h;
    out->data[0] = (uint8_t*) malloc(out_y_pixels);

    int srcStride[1] = {in->width};
    int start = 0;
    int end = in->height;
    int dstStride[1] = {outlink->w};

    int out_height = sws_scale(scale, &input, srcStride, start, end, &out->data[0], dstStride);
    
    int num_channels = 1;

    int in_patch_w = s->patch_width;
    int in_patch_h = s->patch_height;

    int out_patch_w = in_patch_w * s->scale;
    int out_patch_h = in_patch_h * s->scale;
    int in_patch_pixels = in_patch_w * in_patch_h;
    int out_patch_pixels = out_patch_w * out_patch_h;

    uint8_t* lr_patch = (uint8_t*) malloc(in_patch_pixels*num_batches);

    for(int z=0;z<num_batches;z++)
    {
        // int i = batch_i[z];
        char c = current_patch[z+2];    
        int i = c;
        i = i-65;
        int h_idx = i / 6;
        int w_idx = i % 6;
        for(int j = 0;j<in_patch_h;j++)
        {
            memcpy(lr_patch+z*in_patch_pixels+j*in_patch_w, input+h_idx*in_patch_h*inlink->w+w_idx*in_patch_w+j*inlink->w, in_patch_w);
        }
    }

    uint8_t* hr_image = CPytorchModel_forward(s->model, lr_patch, num_batches, num_channels, in_patch_w, in_patch_h);

    for(int z=0;z<num_batches;z++)
    {
        char c = current_patch[z+2];    
        int i = c;
        i = i-65;
        int h_idx = i / 6;
        int w_idx = i % 6;

        for(int j = 0;j<out_patch_h;j++)
        {
            memcpy(out->data[0]+j*outlink->w+w_idx*out_patch_w+h_idx*out_patch_h*outlink->w, hr_image+z*out_patch_pixels+j*out_patch_w,out_patch_w);
        }
    }


    int res = ff_filter_frame(outlink, out);
    av_frame_free(&in);
    free(input);
    free(lr_patch);
    free(current_patch);
    free(hr_image);
    gettimeofday(&etv, NULL);
    
    
    per_frame_time = per_frame_time * i;
    per_frame_time += etv.tv_sec
        *1000+etv.tv_usec/1000-stv.tv_sec
        *1000-stv.tv_usec/1000;
    per_frame_time = per_frame_time/(i+1);
    i += 1;

    return 0;
}



#define OFFSET(x) offsetof(EDSRContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

/*
- name is the option name, keep it simple and lowercase
- description are short, in lowercase, without period, and describe what they
  do, for example "set the foo of the bar"
- offset is the offset of the field in your local context, see the OFFSET()
  macro; the option parser will use that information to fill the fields
  according to the user input
- type is any of AV_OPT_TYPE_* defined in libavutil/opt.h
- default value is an union where you pick the appropriate type; "{.dbl=0.3}",
  "{.i64=0x234}", "{.str=NULL}", ...
- min and max values define the range of available values, inclusive
- flags are AVOption generic flags. See AV_OPT_FLAG_* definitions

*/

// TODO update this accordingly
// TODO is the scale upper bound inclusive or exclusive?
static const AVOption edsr_options[] = {
        { "width", "Frame width.", OFFSET(frame_width), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 16384, .flags = FLAGS },
        { "height", "Frame height.", OFFSET(frame_height), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 16384, .flags = FLAGS },
        //{ "batch_size", "Batch size", OFFSET(batch_size), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 16384, .flags = FLAGS },
        { "patch_width", "Patch width", OFFSET(patch_width), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 16384, .flags = FLAGS },
        { "patch_height", "Patch height", OFFSET(patch_height), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 16384, .flags = FLAGS },
        { "scale", "Upscale ratio (x2, x3, x4)", OFFSET(scale), AV_OPT_TYPE_INT, {.i64 = 2}, 2, 4, .flags = FLAGS },
        //{ "patch_idx", "The index of patches to do SR", OFFSET(patch_idx), AV_OPT_TYPE_STRING },
        { "file_path", "The path to the patch file", OFFSET(file_path), AV_OPT_TYPE_STRING },
        { "model_path", "The path to the trained EDSR model", OFFSET(model_path), AV_OPT_TYPE_STRING },
        { NULL }
};

//AVFILTER_DEFINE_CLASS(edsr);

static const AVClass edsr_class = {
        .class_name       = "edsr",
        .item_name        = av_default_item_name,
        .option           = edsr_options,
        .version          = LIBAVUTIL_VERSION_INT,
        .category         = AV_CLASS_CATEGORY_FILTER,
};

static const AVFilterPad avfilter_vf_edsr_inputs[] = {
        {
                .name = "default",
                .type = AVMEDIA_TYPE_VIDEO,
                .filter_frame = filter_frame,
        },
        { NULL }
};


static const AVFilterPad avfilter_vf_edsr_outputs[] = {
        {
                .name = "default",
                .type = AVMEDIA_TYPE_VIDEO,
                .config_props = config_output,
        },
        { NULL }
};


AVFilter ff_vf_edsr = {
        .name = "edsr",
        .description = NULL_IF_CONFIG_SMALL("Upscales video quality to 2, 3, or 4 times the original size"),
        .init_dict = init_dict,
        .uninit = uninit,
        .priv_size = sizeof(EDSRContext),
        .priv_class = &edsr_class,
        .inputs = avfilter_vf_edsr_inputs,
        .outputs = avfilter_vf_edsr_outputs,
};
