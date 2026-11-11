#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

#include <math.h>

namespace nnet {

template <class res_T, typename CONFIG_T> void fill_zero(hls::stream<res_T> &res) {
    #pragma HLS INLINE
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = 0;
    }
    res.write(res_part);
}

template <class data_T, class res_T, typename CONFIG_T> void fill_data(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    data_T data_part = data.read();
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = data_part[c];
    }
    res.write(res_part);
}

// Vitis-specific override of hls4ml/templates/vivado/nnet_utils/nnet_padding_stream.h:
// each per-pixel loop below now carries #pragma HLS PIPELINE so Vitis HLS can classify
// it for rewind pipelining (fixed, back-to-back latency) instead of leaving it as a
// plain sequential loop whose duration depends on upstream stream fill rate. Mirrors
// the same fix applied to nnet_conv2d_stream.h's conv_2d_buffer_resource_cl.
template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
PadLeft:
    for (int i = 0; i < CONFIG_T::pad_left; i++) {
        #pragma HLS PIPELINE
        fill_zero<res_T, CONFIG_T>(res);
    }

CopyMain:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
        #pragma HLS PIPELINE
        fill_data<data_T, res_T, CONFIG_T>(data, res);
    }

PadRight:
    for (int i = 0; i < CONFIG_T::pad_right; i++) {
        #pragma HLS PIPELINE
        fill_zero<res_T, CONFIG_T>(res);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {

PadTop:
    for (int i = 0; i < CONFIG_T::pad_top; i++) {
    PadTopWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

PadMain:
    for (int i = 0; i < CONFIG_T::in_height; i++) {
    PadLeft:
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            #pragma HLS PIPELINE
            fill_zero<res_T, CONFIG_T>(res);
        }
    CopyMain:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            #pragma HLS PIPELINE
            fill_data<data_T, res_T, CONFIG_T>(data, res);
        }
    PadRight:
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            #pragma HLS PIPELINE
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

PadBottom:
    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
    PadBottomWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE
            fill_zero<res_T, CONFIG_T>(res);
        }
    }
}

} // namespace nnet

#endif
