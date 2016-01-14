#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
__global__ void ThresholdForward(const int n, const Mtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0 ;
  }
}

template <typename Dtype, typename Mtype>
void ThresholdLayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
  Dtype* top_data = top[0]->mutable_gpu_data_base<Dtype>();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<Dtype,Mtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);
INSTANTIATE_LAYER_GPU_FORWARD_FF(ThresholdLayer);


}  // namespace caffe
