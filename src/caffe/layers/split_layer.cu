#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void SplitLayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype, typename Mtype>
void SplitLayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
      const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->gpu_diff_base<Dtype>(), bottom[0]->mutable_gpu_diff_base<Dtype>());
    return;
  }
  caffe_gpu_add<Dtype,Mtype>(count_, top[0]->gpu_diff_base<Dtype>(), top[1]->gpu_diff_base<Dtype>(),
                bottom[0]->mutable_gpu_diff_base<Dtype>());
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff_base<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff_base<Dtype>();
    caffe_gpu_axpy<Dtype,Mtype>(count_, Mtype(1.), top_diff, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
