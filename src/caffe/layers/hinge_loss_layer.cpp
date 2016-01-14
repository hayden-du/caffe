#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, typename Mtype>
void HingeLossLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
  const Dtype* label = bottom[1]->cpu_data_base<Dtype>();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] =
        bottom_diff[i * dim + static_cast<int>(label[i])] * -1 ;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
						      Dtype(0.f), Dtype(1. + bottom_diff[i * dim + j]));
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data_base<Dtype>();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum<Dtype,Mtype>(count, bottom_diff) / num ;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot<Dtype,Mtype>(count, bottom_diff, bottom_diff) / num ;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype, typename Mtype>
void HingeLossLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down, const vector<BlobBase*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
    const Dtype* label = bottom[1]->cpu_data_base<Dtype>();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] =
          bottom_diff[i * dim + static_cast<int>(label[i])] * -1;
    }

    const Mtype loss_weight = top[0]->cpu_diff_base<Dtype>()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal<Dtype,Mtype>(count, Mtype(loss_weight / num), bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal<Dtype,Mtype>(count, Mtype(loss_weight * 2 / num), bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(HingeLossLayer);
REGISTER_LAYER_CLASS(HingeLoss);

}  // namespace caffe
