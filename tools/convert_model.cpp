#include <stdint.h>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

#ifdef CPU_ONLY  // CPU-only Caffe.
int main(int argc, char** argv) {
  std::cout << "Not available in CPU_ONLY mode" << std::endl;
  return 0;
}
#else
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");

int main(int argc, char** argv) {

#ifdef DEBUG
  FLAGS_colorlogtostderr = 0;
  FLAGS_stderrthreshold = 0;
  FLAGS_alsologtostderr = 0;
#else
//  FLAGS_alsologtostderr = 1;
#endif

  ::google::InitGoogleLogging(argv[0]);
  gflags::SetUsageMessage("Converts Caffe model file\n"
        "Usage:\n"
        "    convert_model [FLAGS] INPUT_MODEL OUTPUT_MODEL\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_model");
    return 1;
  }

  map<std::string, shared_ptr<LayerParameter> > layerpar16map;
//  vector<shared_ptr<Blob<float> > > all32blobs;
  vector<Blob<float>*> all32blobs;
  {
    shared_ptr<Net<float,float> > net(new Net<float,float>(FLAGS_model, caffe::TRAIN));
    const string trained_filename(argv[1]);
    net->CopyTrainedLayersFromBinaryProto(trained_filename);

    std::cout << "Reading..." << std::endl;

    const vector<string>& layer_names = net->layer_names();
    for (vector<string>::const_iterator it = layer_names.begin();
        it != layer_names.end(); ++it) {
      const std::string& layer_name = *it;

      std::cout << "Layer: " << layer_name << std::endl;

      shared_ptr<LayerBase> layer = net->layer_by_name(layer_name);
      const vector<shared_ptr<BlobBase> >& blobs = layer->blobs();
//      all32blobs.insert(all32blobs.end(), blobs.begin(), blobs.end());
      for (vector<shared_ptr<BlobBase> >::const_iterator it = blobs.begin();
          it != blobs.end(); ++it) {
        all32blobs.push_back((*it)->instance<float>());
      }

      const LayerParameter& layer_param = layer->layer_param();
      shared_ptr<LayerParameter> layer_param16(new LayerParameter);
      layer_param16->CopyFrom(layer_param);
      layer_param16->clear_blobs();

      for (int i = 0; i < blobs.size(); ++i) {
        BlobProto blob_proto;
        blobs[i]->ToProto(&blob_proto, true);

        BlobProto* blob_proto16 = layer_param16->add_blobs();
        blob_proto16->mutable_shape()->CopyFrom(blob_proto.shape());
        const int data_size = blob_proto.data_size();
        blob_proto16->mutable_half_data()->Reserve(data_size);

        std::cout << "\tBlob " << i << ": " << data_size << std::endl;

        for (int j = 0; j < data_size; ++j) {
          blob_proto16->mutable_half_data()->Add(float16(blob_proto.data(j)).getx());
        }
      }
      layerpar16map[layer_name] = layer_param16;
    }
  }

  {
    shared_ptr<Net<float16,CAFFE_FP16_MTYPE> >
    net16(new Net<float16,CAFFE_FP16_MTYPE>(FLAGS_model, caffe::TRAIN));
    const string trained_filename(argv[1]);
    net16->CopyTrainedLayersFromBinaryProto(trained_filename);

    std::cout  << std::endl << "Writing..." << std::endl;

    NetParameter net_param16;
    net16->ToProto(&net_param16, true);
    net_param16.clear_layer();

//    int all32idx = 0;
    const vector<string>& layer_names = net16->layer_names();
    for (vector<string>::const_iterator it = layer_names.begin();
        it != layer_names.end(); ++it) {
      const std::string& layer_name = *it;
      std::cout << "Layer: " << layer_name << std::endl;
      shared_ptr<LayerParameter> layer_param16 = layerpar16map[layer_name];
      net_param16.add_layer()->CopyFrom(*layer_param16);

// in-place verifier:
//      int blobs_size = layer_param16->blobs_size();
//      for (int j = 0; j < blobs_size; ++j) {
//        const BlobProto& blob_proto16 = layer_param16->blobs(j);
//        Blob<float16> blob16;
//        blob16.FromProto(blob_proto16, true);
//        shared_ptr<Blob<float> > blob32 = all32blobs[all32idx++];
//        int cnt16 = blob16.count();
//        int cnt32 = blob32->count();
//        if (cnt16 != cnt32) {
//          LOG(FATAL) << "Layer " << layer_name << " failed verification: cnt16="
//              << cnt16 << " vs cnt32=" << cnt32;
//        }
//        const float* p32 = blob32->cpu_data();
//        const float16* p16 = blob16.cpu_data();
//        for (int k = 0; k < cnt16; ++k) {
//          float v32 = p32[k];
//          float v16 = (float)p16[k];
//          float diff = fabs(v32 - v16) / std::max(1., fabs(v32));
//          if (diff > 1.e-3) {
//            LOG(WARNING) << "Layer " << layer_name << " failed verification: v16="
//                << v16 << " vs v32=" << v32 << " for blob " << j << ", elem " << k;
//          }
//        }
//      }
    }

    WriteProtoToBinaryFile(net_param16, argv[2]);
  }

  {
    // Reading back the file we just created and verifying its blobs.
    shared_ptr<Net<float16,CAFFE_FP16_MTYPE> >
    net16(new Net<float16,CAFFE_FP16_MTYPE>(FLAGS_model, caffe::TRAIN));
    const string converted_filename(argv[2]);
    net16->CopyTrainedLayersFromBinaryProto(converted_filename);

    std::cout  << std::endl << "Verifying..." << std::endl;

    int all32idx = 0;
    const vector<string>& layer_names = net16->layer_names();
    for (vector<string>::const_iterator it = layer_names.begin();
        it != layer_names.end(); ++it) {
      const std::string& layer_name = *it;

      std::cout << "Layer: " << layer_name << std::endl;

      shared_ptr<LayerBase> layer = net16->layer_by_name(layer_name);
      const vector<shared_ptr<BlobBase> >& blobs = layer->blobs();

      int blobs_size = blobs.size();
      for (int j = 0; j < blobs_size; ++j) {
        Blob<float16>* blob16 = blobs[j]->instance<float16>();
        Blob<float>* blob32 = all32blobs[all32idx++];
        int cnt16 = blob16->count();
        int cnt32 = blob32->count();
        if (cnt16 != cnt32) {
          LOG(FATAL) << "Layer " << layer_name << " failed verification for blob "
              << j << ": cnt16=" << cnt16 << " vs. cnt32=" << cnt32;
        }
        {
          const float* p32 = blob32->cpu_data();
          const float16* p16 = blob16->cpu_data();
          for (int k = 0; k < cnt16; ++k) {
            float v32 = p32[k];
            float v16 = (float)p16[k];
            float diff = fabs(v32 - v16) / std::max(1., fabs(v32));
            if (diff > 1.e-3) {
              LOG(WARNING) << "Layer " << layer_name << " failed data verification: v16="
                  << v16 << " vs. v32=" << v32 << " for blob " << j << ", elem " << k;
            }
          }
        }
        {
          const float* p32 = blob32->cpu_diff();
          const float16* p16 = blob16->cpu_diff();
          for (int k = 0; k < cnt16; ++k) {
            float v32 = p32[k];
            float v16 = (float)p16[k];
            float diff = fabs(v32 - v16) / std::max(1., fabs(v32));
            if (diff > 1.e-3) {
              LOG(WARNING) << "Layer " << layer_name << " failed diff verification: v16="
                  << v16 << " vs. v32=" << v32 << " for blob " << j << ", elem " << k;
            }
          }
        }
      }
    }
  }
  return 0;
}
#endif // CPU_ONLY
