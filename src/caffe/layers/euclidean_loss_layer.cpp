#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Reshape(
  const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data<Dtype>(),
      bottom[1]->cpu_data<Dtype>(),
      diff_.mutable_cpu_data());
  Mtype dot = caffe_cpu_dot<Dtype,Mtype>(count, diff_.cpu_data(), diff_.cpu_data());
  Mtype loss = dot / bottom[0]->num() / 2.;
  top[0]->mutable_cpu_data<Dtype>()[0] = loss;
}

template <typename Dtype, typename Mtype>
void EuclideanLossLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Mtype sign(i == 0 ? 1.f : -1.f);
      const Mtype alpha(sign * top[0]->cpu_diff<Dtype>()[0] / bottom[i]->num());
      caffe_cpu_axpby<Dtype,Mtype>(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Mtype(0),                           // beta
          bottom[i]->mutable_cpu_diff<Dtype>());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
