import torch
import sys
import caffe.io as io
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf.text_format as text_format
from torch.autograd import Variable
from graphviz import Digraph

def make_dot(var):

    node_attr = dict(style = 'filled',
                     shape = 'box',
                     align = 'left',
                     fontsize = '12',
                     ranksep = '0.1',
                     height = '0.2');

    dot = Digraph(node_attr = node_attr, graph_attr = dict(size='12,12'));
    seen = set();

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), 'Tensor' + size_to_str(var.size()), fillcolor='orange');

            elif hasattr(var, 'variable'):
                u = var.variable;
                dot.node(str(id(var)), type(u).__name__ + size_to_str(u.size()), fillcolor='lightblue');

            else:

                dot.node(str(id(var)), type(var).__name__);

            seen.add(var);
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)));
                        add_nodes(u[0]);

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)));
                    add_nodes(t);

    add_nodes(var.grad_fn);
    return dot;

def init_name_dict():
    name_dict = dict();
    name_dict['ConvNdBackward'] = 'Convolution';
    name_dict['ThresholdBackward'] = 'ReLU';
    name_dict['LeakyReLUBackward'] = 'ReLU';
    name_dict['AddmmBackward'] = 'InnerProduct';
    name_dict['LogSoftmaxBackward'] = 'Softmax';
    name_dict['SoftmaxBackward'] = 'Softmax';
    name_dict['MaxPool2dBackward'] = 'Pooling';
    name_dict['AvgPool2dBackward'] = 'Pooling';
    name_dict['ViewBackward'] = 'Reshape';
    name_dict['IndexBackward'] = 'Index';
    name_dict['ConcatBackward'] = 'Concat';
    name_dict['AddBackward'] = 'Eltwise';
    name_dict['MulBackward'] = 'Eltwise';
    name_dict['CmaxBackward'] = 'Eltwise';
    name_dict['BatchNormBackward'] = 'BatchNorm';
    name_dict['TanhBackward'] = 'TanH';
    name_dict['DropoutBackward'] = 'Dropout';
    return name_dict;

def to_caffe(output, input_shapes, filename, deletes):

    m_dot = make_dot(output);
    m_dot.save(filename + '.dot');

    type_name_dict = init_name_dict();

    type_list = list();
    node_set = set();
    node_list = list();
    parent_list = list();
    child_list = list();

    net_layers = list();
    final_layers = list();
    final_tops = set();

    def convolution_params(var, layer, flag_deconv):
        layer.convolution_param.stride_h = var.stride[0];
        layer.convolution_param.stride_w = var.stride[1];
        layer.convolution_param.pad_h = var.padding[0];
        layer.convolution_param.pad_w = var.padding[1];

        weight_var = var.next_functions[1][0];
        bias_var = var.next_functions[2][0];

        layer.convolution_param.num_output = weight_var.variable.size()[int(flag_deconv)];
        layer.convolution_param.kernel_h = weight_var.variable.size()[2];
        layer.convolution_param.kernel_w = weight_var.variable.size()[3];

        layer.convolution_param.group = var.groups;

        if bias_var is None:
            bias_array = torch.zeros(layer.convolution_param.num_output).numpy();
        else:
            bias_array = bias_var.variable.data.numpy();

        weight_array = weight_var.variable.data.numpy();
        layer.blobs.extend([io.array_to_blobproto(weight_array)]);
        layer.blobs.extend([io.array_to_blobproto(bias_array)]);

    def fully_connected_params(var, layer):
        layer.inner_product_param.num_output = var.add_matrix_size[0];
        bias_var = var.next_functions[0][0];
        weight_var = var.next_functions[2][0].next_functions[0][0];

        weight_array = weight_var.variable.data.numpy();
        layer.blobs.extend([io.array_to_blobproto(weight_array)]);

        bias_array = bias_var.variable.data.numpy();
        layer.blobs.extend([io.array_to_blobproto(bias_array)]);

    def pooling_params(var, method, layer):
        if method == 'MAX':
            layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX;
        elif method == 'AVE':
            layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE;

        layer.pooling_param.pad_h = var.padding[0];
        layer.pooling_param.pad_w = var.padding[1];
        layer.pooling_param.kernel_h = var.kernel_size[0];
        layer.pooling_param.kernel_w = var.kernel_size[1];
        layer.pooling_param.stride_h = var.stride[0];
        layer.pooling_param.stride_w = var.stride[1];

    def slice_params(layer, tops):
        child_var = node_list[tops[0]];
        for i in range(len(child_var.index)):
            if child_var.index[i].start is not None:
                layer.slice_param.axis = i;
                break;
        for i in range(1, len(tops)):
            child_var = node_list[tops[i]];
            start = child_var.index[layer.slice_param.axis].start;
            layer.slice_param.slice_point.extend([start]);

    def entrywise_params(layer, operation):
        if operation == "PROD":
            layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.PROD;
        if operation == "SUM":
            layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM;
        if operation == "MAX":
            layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.MAX;

    def batch_norm_params(var, layer):
        layer.batch_norm_param.moving_average_fraction = var.momentum;
        layer.batch_norm_param.eps = var.eps;

        layer.blobs.extend([io.array_to_blobproto(var.running_mean.numpy())]);
        layer.blobs.extend([io.array_to_blobproto(var.running_var.numpy())]);
        layer.blobs.extend([io.array_to_blobproto(torch.ones(1).numpy())]);

    def is_convolution(var):
        return type(var).__name__ == 'ConvNdBackward';

    def is_rectified_linear_unit(var):
        return type(var).__name__ == 'ThresholdBackward';

    def is_leaky_relu(var):
        return type(var).__name__ == 'LeakyReLUBackward';

    def is_fully_connected(var):
        return type(var).__name__ == 'AddmmBackward';

    def is_softmax(var):
        return type(var).__name__ == 'SoftmaxBackward';

    def is_tanh(var):
        return type(var).__name__ == 'TanhBackward';

    def is_log_softmax(var):
        return type(var).__name__ == 'LogSoftmaxBackward';

    def is_max_pooling(var):
        return type(var).__name__ == 'MaxPool2dBackward';

    def is_average_pooling(var):
        return type(var).__name__ == 'AvgPool2dBackward';

    def is_index(var):
        return type(var).__name__ == 'IndexBackward';

    def is_concatenate(var):
        return type(var).__name__ == 'ConcatBackward';

    def is_reshape(var):
        return type(var).__name__ == 'ViewBackward';

    def is_entrywise_product(var):
        return type(var).__name__ == 'MulBackward';

    def is_entrywise_sum(var):
        return type(var).__name__ == 'AddBackward';

    def is_entrywise_max(var):
        return type(var).__name__ == 'CmaxBackward';

    def is_batch_norm(var):
        return type(var).__name__ == 'BatchNormBackward';

    def is_dropout(var):
        return type(var).__name__ == 'DropoutBackward';

    def set_tops(layer, tops):
        if not tops:
            layer.top.extend(['chance']);
        else:
            layer.top.extend([layer.name]);

    def set_bottoms(layer, bottoms, index):
        if not bottoms:
            last_backward = node_list[index].next_functions[0][0];

            assert hasattr(last_backward, 'variable');

            flag = False;

            for i, shape in enumerate(input_shapes):
                if last_backward.variable.size() == shape:
                    layer.bottom.extend(['crystal' + str(i)]);
                    flag = True;
            
            assert flag;

        for bottom in bottoms:
            if type_list[bottom] == 'BatchNorm':
                layer.bottom.extend(['Scale' + str(bottom)]);
            else:
                layer.bottom.extend([type_list[bottom] + str(bottom)]);

    def parse_convolution(var, index, bottoms, tops, flag_deconv):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        convolution_params(var, layer, flag_deconv);
        net_layers.append(layer);

    def parse_fully_connected(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        fully_connected_params(var, layer);
        net_layers.append(layer);

    def parse_log_softmax(var, index, bottoms, tops):
        softmax_layer = caffe_pb2.LayerParameter();
        softmax_layer.name = type_list[index] + str(index);
        softmax_layer.type = type_list[index];

        log_layer = caffe_pb2.LayerParameter();
        log_layer.name = 'Log' + str(index);
        log_layer.type = 'Log';

        softmax_layer.top.extend([softmax_layer.name]);
        log_layer.bottom.extend([softmax_layer.name]);

        set_tops(log_layer, tops);
        set_bottoms(softmax_layer, bottoms, index);

        net_layers.append(log_layer);
        net_layers.append(softmax_layer);

    def parse_tanh(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        net_layers.append(layer);

    def parse_softmax(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        net_layers.append(layer);

    def parse_pooling(var, index, bottoms, tops, method):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        pooling_params(var, method, layer);
        net_layers.append(layer);

    def parse_rectified_linear_unit(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        net_layers.append(layer);

    def parse_leaky_relu(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        layer.relu_param.negative_slope = var.additional_args[0];
        net_layers.append(layer);

    def parse_dropout(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        layer.dropout_param.dropout_ratio = 1;
        net_layers.append(layer);

    def parse_concatenate(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        layer.concat_param.axis = var.dim;
        net_layers.append(layer);

    def parse_index(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        net_layers.append(layer);

    def parse_reshape(var, index, bottoms, tops):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);
        layer.reshape_param.shape.dim.extend(var.new_sizes);
        net_layers.append(layer);

    def parse_entrywise_operation(var, index, bottoms, tops, operation):
        layer = caffe_pb2.LayerParameter();
        layer.name = type_list[index] + str(index);
        layer.type = type_list[index];

        set_tops(layer, tops);
        set_bottoms(layer, bottoms, index);

        entrywise_params(layer, operation);
        net_layers.append(layer);

    def generate_scale(var, index):
        scale_layer = caffe_pb2.LayerParameter();
        scale_layer.name = 'Scale' + str(index);
        scale_layer.type = 'Scale';
        scale_layer.bottom.extend(['BatchNorm' + str(index)]);
        scale_layer.scale_param.bias_term = True;

        scale_array = var.next_functions[1][0].variable.data.numpy();
        bias_array = var.next_functions[2][0].variable.data.numpy();

        scale_layer.blobs.extend([io.array_to_blobproto(scale_array)]);
        scale_layer.blobs.extend([io.array_to_blobproto(bias_array)])
        return scale_layer;

    def parse_batch_norm(var, index, bottoms, tops):
        batch_norm_layer = caffe_pb2.LayerParameter();
        batch_norm_layer.name = type_list[index] + str(index);
        batch_norm_layer.type = type_list[index];

        batch_norm_layer.top.extend([batch_norm_layer.name]);
        batch_norm_params(var, batch_norm_layer);

        scale_layer = generate_scale(var, index);
        set_bottoms(batch_norm_layer, bottoms, index);
        set_tops(scale_layer, tops);

        net_layers.append(scale_layer);
        net_layers.append(batch_norm_layer);

    def collect_var():
        list_index = 0;
        while list_index < len(node_list):
            var = node_list[list_index];
            type_list.append(type_name_dict[type(var).__name__]);

            for u in var.next_functions:
                if type(u[0]).__name__ not in type_name_dict:
                    continue;

                if u[0] not in node_set:
                    node_set.add(u[0]);
                    node_list.append(u[0]);
                    parent_list.append(list());
                    child_list.append(list());

                parent = node_list.index(u[0]);
                parent_list[list_index].append(parent);
                child_list[parent].append(list_index);
            
            list_index += 1;

    def detect_deconv():
        for list_index in range(len(node_list)):
            var = node_list[list_index];

            if not is_convolution(var):
                continue;

            parent = var.next_functions[0][0];

            if is_rectified_linear_unit(parent):
                type_list[list_index] = 'Deconvolution';

    def convert_var():
        for list_index in range(len(node_list)):
            var = node_list[list_index];
            bottoms = parent_list[list_index];
            tops = child_list[list_index];

            if is_convolution(var):
                flag_deconv = type_list[list_index] == 'Deconvolution';
                parse_convolution(var, list_index, bottoms, tops, flag_deconv);

            elif is_fully_connected(var):
                parse_fully_connected(var, list_index, bottoms, tops);

            elif is_tanh(var):
                parse_tanh(var, list_index, bottoms, tops);

            elif is_softmax(var):
                parse_softmax(var, list_index, bottoms, tops);

            elif is_log_softmax(var):
                parse_log_softmax(var, list_index, bottoms, tops);

            elif is_max_pooling(var):
                parse_pooling(var, list_index, bottoms, tops, 'MAX');

            elif is_average_pooling(var):
                parse_pooling(var, list_index, bottoms, tops, 'AVE');

            elif is_rectified_linear_unit(var):
                parse_rectified_linear_unit(var, list_index, bottoms, tops);

            elif is_leaky_relu(var):
                parse_leaky_relu(var, list_index, bottoms, tops);

            elif is_index(var):
                parse_index(var, list_index, bottoms, tops);

            elif is_concatenate(var):
                parse_concatenate(var, list_index, bottoms, tops);

            elif is_reshape(var):
                parse_reshape(var, list_index, bottoms, tops);

            elif is_entrywise_product(var):
                parse_entrywise_operation(var, list_index, bottoms, tops, "PROD");

            elif is_entrywise_sum(var):
                parse_entrywise_operation(var, list_index, bottoms, tops, "SUM");

            elif is_entrywise_max(var):
                parse_entrywise_operation(var, list_index, bottoms, tops, "MAX");

            elif is_batch_norm(var):
                parse_batch_norm(var, list_index, bottoms, tops);

            elif is_dropout(var):
                parse_dropout(var, list_index, bottoms, tops);

    def get_layer_name(list_index):
        return type_list[list_index] + str(list_index);

    def generate_slice(var, list_index, tops):
        slice_layer = caffe_pb2.LayerParameter();
        slice_layer.name = 'Slice' + str(list_index);
        slice_layer.type = 'Slice';

        if type_list[list_index] == 'BatchNorm':
            slice_layer.bottom.extend(['Scale' + str(list_index)]);
        else:
            slice_layer.bottom.extend([type_list[list_index] + str(list_index)]);

        slice_layer.top.extend([get_layer_name(top) for top in tops]);

        slice_params(slice_layer, tops);
        return slice_layer;

    def reconnect_slice():
        slice_layers = list();

        for list_index in range(len(node_list)):
            tops = child_list[list_index];
            if tops and is_index(node_list[tops[0]]):
                var = node_list[list_index];
                slice_layers.append(generate_slice(var, list_index, tops));

        for layer in slice_layers:
            net_layers.append(layer);

    def delete_index_layers():
        to_delete = list();
        for layer in net_layers:
            if layer.type == 'Index':
                to_delete.append(layer);
        for layer in to_delete:
            net_layers.remove(layer);

    def gather_layer_indices():
        layer_index_dict = dict();
        for i in range(len(net_layers)):
            layer_index_dict[net_layers[i].name] = i;
        return layer_index_dict;

    def delete_additional():
        if not deletes:
            return;

        layer_indices = gather_layer_indices();

        assert is_concatenate(node_list[0]);

        root_layer_name = get_layer_name(0);

        additional_layers = list();

        additional_layers.append(net_layers[layer_indices[root_layer_name]]);

        for index, view_index in enumerate(parent_list[0]):
            view_layer_name = get_layer_name(view_index);
            additional_layers.append(net_layers[layer_indices[view_layer_name]])

            margin_index = parent_list[view_index][0];

            if type_list[margin_index] == 'BatchNorm':
                margin_layer_name = 'Scale' + str(margin_index);
            else:
                margin_layer_name = get_layer_name(margin_index);

            margin_layer = net_layers[layer_indices[margin_layer_name]];

            margin_layer.top.pop()
            margin_layer.top.extend(['chance' + str(index)]);

        for layer in additional_layers:
            net_layers.remove(layer);

    def generate_input(net_definition):
        crystal_names = ['crystal' + str(i) for i in range(len(input_shapes))];

        net_definition.input.extend(crystal_names);

        crystal_dims = list();

        for i in range(len(input_shapes)):
            caffe_shape = caffe_pb2.BlobShape();
            caffe_shape.dim.extend(list(input_shapes[i]));

            crystal_dims.append(caffe_shape);

        net_definition.input_shape.extend(crystal_dims);

        for name in crystal_names:
            final_tops.add(name);

    def rearrange_layers():
        while True:
            flag = False;

            for layer in net_layers:
                qualifies = True;
                
                for parent in layer.bottom:
                    if parent not in final_tops:
                        qualifies = False;
                        break;
                
                if qualifies:
                    final_layers.append(layer);
                    for child in layer.top:
                        final_tops.add(child);
                    net_layers.remove(layer);

                    flag = True;
                    break;

            if not flag:
                break;

    def save_network():
        net_definition = caffe_pb2.NetParameter();
        generate_input(net_definition);

        net_weights = caffe_pb2.NetParameter();
        net_weights.CopyFrom(net_definition);

        rearrange_layers();

        for layer in final_layers:
            net_weights.layer.extend([layer])
            without_weights = caffe_pb2.LayerParameter()
            without_weights.CopyFrom(layer)
            del without_weights.blobs[:]
            net_definition.layer.extend([without_weights])

        with open(filename + '.prototxt', 'w') as f:
            f.write(text_format.MessageToString(net_definition));
        with open(filename + '.caffemodel', 'w') as f:
            f.write(net_weights.SerializeToString())

    node_set.add(output.grad_fn);
    node_list.append(output.grad_fn);
    parent_list.append(list());
    child_list.append(list());

    collect_var();
    detect_deconv();
    convert_var();
    reconnect_slice();
    delete_index_layers();
    delete_additional();
    save_network();

def convert(model, input_shapes, filename):
    model.cpu();
    model.eval();

    input_shapes = [(1, shape[0], shape[1], shape[2]) for shape in input_shapes];

    input_tensors = [torch.randn(shape) for shape in input_shapes];

    input_variables = [Variable(tensor, requires_grad = False) for tensor in input_tensors];

    outputs = model(*input_variables);

    if isinstance(outputs, tuple):
        flattened = [output.view(output.size(0), -1) for output in outputs];
        for flat in flattened:
            print 'flattened', flat.size();
        
        merge = torch.cat(tuple(flattened), 1);
        to_caffe(merge, input_shapes, filename, True);
    else:
        to_caffe(outputs, input_shapes, filename, False);