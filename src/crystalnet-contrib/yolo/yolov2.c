#include <stddef.h>
#include <stdio.h>

#include <crystalnet-contrib/yolo/conv_layer.h>
#include <crystalnet-contrib/yolo/pool_layer.h>
#include <crystalnet-contrib/yolo/region_layer.h>
#include <crystalnet-contrib/yolo/reorg_layer.h>
#include <crystalnet-contrib/yolo/route_layer.h>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet-ext.h>

const uint32_t yolov2_input_size = 416;

s_model_t *yolov2(context_t *ctx)
{
    const shape_t *input_shape =
        mk_shape(ctx, 3, 3, yolov2_input_size, yolov2_input_size);

    s_layer_t *max_pool2 = pool(2, 2);
    s_layer_t *reorg = make_reorg_layer();
    s_layer_t *region = make_region_layer(13, 13, 5, 80, 4);

    symbol x = var(ctx, input_shape);
    symbol l16 = transform_all(  //
        ctx,                     //
        (p_layer_t[]){
            conv(32, 3, 1, 1),   // 0
            max_pool2,           // 1
            conv(64, 3, 1, 1),   // 2
            max_pool2,           // 3
            conv(128, 3, 1, 1),  // 4
            conv(64, 1, 1, 0),   // 5
            conv(128, 3, 1, 1),  // 6
            max_pool2,           // 7
            conv(256, 3, 1, 1),  // 8
            conv(128, 1, 1, 0),  // 9
            conv(256, 3, 1, 1),  // 10
            max_pool2,           // 11
            conv(512, 3, 1, 1),  // 12
            conv(256, 1, 1, 0),  // 13
            conv(512, 3, 1, 1),  // 14
            conv(256, 1, 1, 0),  // 15
            conv(512, 3, 1, 1),  // 16
            NULL,
        },
        x);

    symbol l24 = transform_all(  //
        ctx,
        (p_layer_t[]){
            max_pool2,            // 17
            conv(1024, 3, 1, 1),  // 18
            conv(512, 1, 1, 0),   // 19
            conv(1024, 3, 1, 1),  // 20
            conv(512, 1, 1, 0),   // 21
            conv(1024, 3, 1, 1),  // 22
            conv(1024, 3, 1, 1),  // 23
            conv(1024, 3, 1, 1),  // 24
            NULL,                 // end
        },
        l16);

    symbol l25 = route_1(ctx, l16);

    symbol l27 = transform_all(  //
        ctx,                     //
        (p_layer_t[]){
            conv(64, 1, 1, 0),  // 26
            reorg,              // 27
            NULL,               // end
        },
        l25);

    symbol l28 = route_2(ctx, l27, l24);

    symbol y = transform_all(  //
        ctx,
        (p_layer_t[]){
            conv(1024, 3, 1, 1),            // 29
            conv_linear_act(425, 1, 1, 0),  // 30
            region,                         // 31
            NULL,                           // end
        },
        l28);

    return make_s_model(ctx, x, y);
}
