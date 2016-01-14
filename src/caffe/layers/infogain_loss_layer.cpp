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
void InfogainLossLayer<Dtype,Mtype>::LayerSetUp(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  LossLayer<Dtype,Mtype>::LayerSetUp(bottom, top);
  if (bottom.size() < 3) {
    CHECK(this->layer_param_.infogain_loss_param().has_source())
        << "Infogain matrix source must be specified.";
    BlobProto blob_proto;
    ReadProtoFromBinaryFile(
      this->layer_param_.infogain_loss_param().source(), &blob_proto);
    infogain_.FromProto(blob_proto);
  }
}

template <typename Dtype, typename Mtype>
void InfogainLossLayer<Dtype,Mtype>::Reshape(
    const vector<BlobBase*>& bottom, const vector<BlobBase*>& top) {
  LossLayer<Dtype,Mtype>::Reshape(bottom, top);
  BlobBase* infogain = NULL;
  if (bottom.size() < 3) {
    infogain = &infogain_;
  } else {
    infogain = bottom[2];
  }
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  CHECK_EQ(infogain->num(), 1);
  CHECK_EQ(infogain->channels(), 1);
  CHECK_EQ(infogain->height(), dim);
  CHECK_EQ(infogain->width(), dim);
}


template <typename Dtype, typename Mtype>
void InfogainLossLayer<Dtype,Mtype>::Forward_cpu(const vector<BlobBase*>& bottom,
    const vector<BlobBase*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
  const Dtype* bottom_label = bottom[1]->cpu_data_base<Dtype>();
  const Dtype* infogain_mat = NULL;
  if (bottom.size() < 3) {
    infogain_mat = infogain_.cpu_data();
  } else {
    infogain_mat = bottom[2]->cpu_data_base<Dtype>();
  }
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Mtype loss(0.f);
  for (int i = 0; i < num; ++i) {
    int label = bottom_label[i];
    for (int j = 0; j < dim; ++j) {
      Mtype prob = std::max(bottom_data[i * dim + j],
          choose<Dtype>(Dtype(kLOG_THRESHOLD), minDtype<Dtype>()));
      loss -= infogain_mat[label * dim + j] * log(prob);
    }
  }
  top[0]->mutable_cpu_data_base<Dtype>()[0] = loss / num;
}

template <typename Dtype, typename Mtype>
void InfogainLossLayer<Dtype,Mtype>::Backward_cpu(const vector<BlobBase*>& top,
    const vector<bool>& propagate_down,
    const vector<BlobBase*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to infogain inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data_base<Dtype>();
    const Dtype* bottom_label = bottom[1]->cpu_data_base<Dtype>();
    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3) {
      infogain_mat = infogain_.cpu_data();
    } else {
      infogain_mat = bottom[2]->cpu_data_base<Dtype>();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff_base<Dtype>();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    const Mtype scale = - top[0]->cpu_diff_base<Dtype>()[0] / num ;
    for (int i = 0; i < num; ++i) {
      const int label = static_cast<int>(bottom_label[i]);
      for (int j = 0; j < dim; ++j) {
        Mtype prob = std::max(bottom_data[i * dim + j],
            choose<Dtype>(Dtype(kLOG_THRESHOLD), minDtype<Dtype>()));
        bottom_diff[i * dim + j] = scale * infogain_mat[label * dim + j] / prob ;
      }
    }
  }
}

INSTANTIATE_CLASS(InfogainLossLayer);
REGISTER_LAYER_CLASS(InfogainLoss);
}  // namespace caffe
