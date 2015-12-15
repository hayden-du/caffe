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
void MultinomialLogisticLossLayer<Dtype,Mtype>::Reshape(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype, typename Mtype>
void MultinomialLogisticLossLayer<Dtype,Mtype>::Forward_cpu(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data<Dtype>();
  const Dtype* bottom_label = bottom[1]->cpu_data<Dtype>();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Mtype loss(0.f);
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Mtype prob = std::max(
        bottom_data[i * dim + label],
        choose<Dtype>(Dtype(kLOG_THRESHOLD), minDtype<Dtype>()));
    loss -= log(prob);
  }
  top[0]->mutable_cpu_data<Dtype>()[0] = loss / num;
}

template <typename Dtype, typename Mtype>
void MultinomialLogisticLossLayer<Dtype,Mtype>::Backward_cpu(
    const vector<BlobBase*>& top, const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data<Dtype>();
    const Dtype* bottom_label = bottom[1]->cpu_data<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff<Dtype>();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), typedConsts<Dtype>::zero, bottom_diff);
    const Mtype scale = - top[0]->cpu_diff<Dtype>()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Mtype prob = std::max(
          bottom_data[i * dim + label],
          choose<Dtype>(Dtype(kLOG_THRESHOLD), minDtype<Dtype>()));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
}

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticLoss);

}  // namespace caffe
