#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void AbsValLayer<Dtype,Mtype>::Forward_gpu(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data_base<Dtype>();
  caffe_gpu_abs<Dtype,Mtype>(count, bottom[0]->gpu_data_base<Dtype>(), top_data);
}

template <typename Dtype, typename Mtype>
void AbsValLayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff_base<Dtype>();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data_base<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff_base<Dtype>();
    caffe_gpu_sign<Dtype,Mtype>(count, bottom_data, bottom_diff);
    caffe_gpu_mul<Dtype,Mtype>(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);


}  // namespace caffe
