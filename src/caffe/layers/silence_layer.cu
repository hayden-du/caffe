#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void SilenceLayer<Dtype,Mtype>::Forward_gpu(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
  // Do nothing.
}

template <typename Dtype, typename Mtype>
void SilenceLayer<Dtype,Mtype>::Backward_gpu(const vector<BlobBase*>& top,
      const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set<Dtype,Mtype>(bottom[i]->count(), Mtype(0),
                    bottom[i]->mutable_gpu_diff_base<Dtype>());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
