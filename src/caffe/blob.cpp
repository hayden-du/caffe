#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void BlobBase::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

void BlobBase::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * dtsize()));
    diff_.reset(new SyncedMemory(capacity_ * dtsize()));
  }
}

void BlobBase::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

void BlobBase::ReshapeLike(const BlobBase& other) {
  Reshape(other.shape());
}

BlobBase::BlobBase(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

BlobBase::BlobBase(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

const int* BlobBase::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

void BlobBase::ShareData(const BlobBase& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

void BlobBase::ShareDiff(const BlobBase& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template<> void BlobBase::Update<unsigned int,unsigned int>() { NOT_IMPLEMENTED; }
template<> void BlobBase::Update<int,int>() { NOT_IMPLEMENTED; }

template <typename Dtype, typename Mtype>
void BlobBase::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy(count_, typedConsts<Mtype>::minus_one,
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy(count_, typedConsts<Mtype>::minus_one,
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template<> unsigned int BlobBase::asum_data<unsigned int,unsigned int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int BlobBase::asum_data<int,int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype, typename Mtype>
Mtype BlobBase::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum<Dtype,Mtype>(count_, (const Dtype*)data_->cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Mtype asum;
    caffe_gpu_asum<Dtype,Mtype>(count_, (const Dtype*)data_->gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template<> unsigned int BlobBase::asum_diff<unsigned int,unsigned int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int BlobBase::asum_diff<int,int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype, typename Mtype>
Mtype BlobBase::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum<Dtype,Mtype>(count_, (const Dtype*)diff_->cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Mtype asum;
    caffe_gpu_asum<Dtype,Mtype>(count_, (const Dtype*)diff_->gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template<> unsigned int BlobBase::sumsq_data<unsigned int,unsigned int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int BlobBase::sumsq_data<int,int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype, typename Mtype>
Mtype BlobBase::sumsq_data() const {
  Mtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = (const Dtype*)data_->cpu_data();
    sumsq = caffe_cpu_dot<Dtype,Mtype>(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = (const Dtype*)data_->gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template<> unsigned int BlobBase::sumsq_diff<unsigned int,unsigned int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template<> int BlobBase::sumsq_diff<int,int>() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype, typename Mtype>
Mtype BlobBase::sumsq_diff() const {
  Mtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = (const Dtype*)diff_->cpu_data();
    sumsq = caffe_cpu_dot<Dtype,Mtype>(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = (const Dtype*)diff_->gpu_data();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template<> void BlobBase::scale_data<unsigned int,unsigned int>(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void BlobBase::scale_data<int,int>(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype, typename Mtype>
void BlobBase::scale_data(Mtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = (const Dtype*)data_->mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = (const Dtype*)data_->mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template<> void BlobBase::scale_diff<unsigned int,unsigned int>(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template<> void BlobBase::scale_diff<int,int>(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype, typename Mtype>
void BlobBase::scale_diff(Mtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = (const Dtype*)diff_->mutable_cpu_data();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = (const Dtype*)diff_->mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

bool BlobBase::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else if (proto.data_size() > 0) {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  } else if (proto.half_data_size() > 0) {
    CHECK_EQ(count_, proto.half_data_size());
    for (int i = 0; i < count_; ++i) {
      float16 h;
      h.setx((unsigned short) proto.half_data(i));
      data_vec[i] = (Dtype) h;
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  } else if (proto.half_diff_size() > 0) {
    CHECK_EQ(count_, proto.half_diff_size());
    for (int i = 0; i < count_; ++i) {
      float16 h;
      h.setx((unsigned short) proto.half_diff(i));
      data_vec[i] = (Dtype) h;
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

#ifndef CPU_ONLY

template <>
void Blob<float16>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float16* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_half_data(data_vec[i].getx());
  }
  if (write_diff) {
    const float16* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_half_diff(float16(diff_vec[i]).getx());
    }
  }
}

template <>
void Blob<float16>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  if (proto.data_size() > 0) {
    CHECK_EQ(count_, proto.data_size());
    float16* data_vec = mutable_cpu_data();
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    } 
  } else if (proto.half_data_size() > 0) {
    float16* data_vec = mutable_cpu_data();
    CHECK_EQ(count_, proto.half_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i].setx(proto.half_data(i));
    }
  }
  if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    float16* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.half_diff(i);
    }
  }
  else if (proto.half_diff_size() > 0) {
    CHECK_EQ(count_, proto.half_diff_size());
    float16* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i].setx(proto.half_diff(i));
    }
  }
}

#endif // CPU_ONLY

INSTANTIATE_CLASS1(Blob);
//INSTANTIATE_MEMBER_VOID_TEMPLATE(BlobBase,Update);
//INSTANTIATE_MEMBER_TEMPLATE_CONST(Blob,asum_data);
//INSTANTIATE_MEMBER_TEMPLATE_CONST(Blob,asum_diff);
//INSTANTIATE_MEMBER_TEMPLATE_CONST(Blob,sumsq_data);
//INSTANTIATE_MEMBER_TEMPLATE_CONST(Blob,sumsq_diff);
//INSTANTIATE_MEMBER_VOID_TEMPLATE_1(Blob,scale_data);
//INSTANTIATE_MEMBER_VOID_TEMPLATE_1(Blob,scale_diff);

// we need full matrix of instantiations for blob
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe
