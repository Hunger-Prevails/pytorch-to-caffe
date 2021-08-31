import torch
import pytorch_converter as pc
from resnet import resnet18

file_path = '/home/sensetime/Documents/Pytorch-Caffe';
model_path = file_path + '/pytorch_models';

resnet_models = dict();
resnet_models[18] = '/resnet18.pth';
resnet_models[34] = '/resnet34.pth';
resnet_models[50] = '/resnet50.pth';
resnet_models[101] = '/resnet101.pth';
resnet_models[152] = '/resnet152.pth';

resnet_model = resnet18();
model_params = torch.load(model_path + resnet_models[18]);
resnet_model.load_state_dict(model_params);

input_shapes = [(3, 224, 224)]
pc.convert(resnet_model, input_shapes, 'resnet');