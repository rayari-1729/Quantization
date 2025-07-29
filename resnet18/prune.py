import torch
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from projects.mmdet3d_plugin.custom_utils import convert_tensor_format, OptimizedLinear, BatchGridSampleFunction

@ATTENTION.register_module()
class MultiScaleDeformableAttention_TSA_Optimized(nn.Module):
    def __init__(self, embed_dims, num_bev_queue, num_heads, num_levels, num_points, im2col_step, dim_per_head, grid_sample_mode="nearest", use_optimized_linear = True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_bev_queue = num_bev_queue
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.use_optimized_linear = use_optimized_linear
        self.grid_sample_mode = grid_sample_mode

        # Define layers for sampling offsets and attention weights
        if self.use_optimized_linear:
            self.sampling_offsets = OptimizedLinear(self.embed_dims*self.num_bev_queue, self.num_bev_queue*self.num_heads * self.num_levels * self.num_points * 2, input_layout = 'nhwc', output_layout = 'nchw')
            self.attention_weights = OptimizedLinear(self.embed_dims*self.num_bev_queue, self.num_bev_queue*self.num_heads * self.num_levels * self.num_points, input_layout = 'nhwc', output_layout = 'nchw')
        else:
            self.sampling_offsets = nn.Linear(self.embed_dims*self.num_bev_queue, self.num_bev_queue*self.num_heads * self.num_levels * self.num_points * 2)
            self.attention_weights = nn.Linear(self.embed_dims*self.num_bev_queue, self.num_bev_queue*self.num_heads * self.num_levels * self.num_points)
        self.init_weights()

    def init_weights(self):
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        if self.use_optimized_linear:
            constant_init(self.sampling_offsets.layer, 0.)
            self.sampling_offsets.layer.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights.layer, val=0., bias=0.)
        else:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)

    def forward(self, value, query, spatial_shapes, level_start_index, reference_points, key_padding_mask=None):
        bs,  num_query1, num_query2, _ = query.shape
        _, _, num_value1, num_value2= value.shape

        num_query = num_query1 * num_query2
        num_value = num_value1 * num_value2

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)


        sampling_offsets = self.sampling_offsets(query).contiguous()
        attention_weights = self.attention_weights(query).contiguous()
        
        if not self.use_optimized_linear:
            sampling_offsets = convert_tensor_format(sampling_offsets, 'nchw')
            attention_weights = convert_tensor_format(attention_weights, 'nchw')

        if bs > 1:
            sampling_offsets = sampling_offsets.split(1, 0)
            sampling_offsets = [i.view(self.num_heads, 2, self.num_points*self.num_bev_queue*self.num_levels, num_query) for i in sampling_offsets]
            sampling_offsets = [i.transpose(1, 0).contiguous() for i in sampling_offsets]
            sampling_offsets = torch.cat(sampling_offsets, dim=0)
        else:
            sampling_offsets = sampling_offsets.view(bs * self.num_heads, 2, self.num_points*self.num_bev_queue*self.num_levels, num_query)
            sampling_offsets = sampling_offsets.transpose(1, 0).contiguous()

        sampling_offsets = sampling_offsets.view( bs * 2, self.num_heads* self.num_points, -1, num_query)

        if bs>1:
            attention_weights = attention_weights.split(1, 0)
            attention_weights = [i.view(
               self.num_heads, self.num_bev_queue, self.num_levels * self.num_points, num_value).transpose(1, 0).contiguous().view(self.num_bev_queue*self.num_heads, 1, self.num_levels * self.num_points, num_value).softmax(2).transpose(2, 3) for i in attention_weights]
            attention_weights = torch.cat(attention_weights, dim=0)
        else:
            attention_weights = attention_weights.view(
                bs * self.num_heads, self.num_bev_queue, self.num_levels * self.num_points, num_value)

            attention_weights = attention_weights.split(1, dim=1)
            attention_weights = torch.cat(attention_weights, axis=0).softmax(2).transpose(2, 3)

        # Calculate sampling locations based on reference points and offsets
        sampling_locations = self.calculate_sampling_locations(reference_points, sampling_offsets, spatial_shapes)

        # Apply the multi-scale deformable attention function
        output = self.apply_deformable_attention(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

        return output

    def calculate_sampling_locations(self, reference_points, sampling_offsets, spatial_shapes):
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
           
            offset_normalizer=1/offset_normalizer[None,None,:,:]
            reference_points = reference_points[:,:,:,:]*2 - 1
            offset_normalizer = 2*offset_normalizer
            sampling_locations = (reference_points.permute(0, 2, 3, 1)) + (sampling_offsets *  offset_normalizer.transpose(3, 2))
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead.')
        return sampling_locations

    def apply_deformable_attention(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        output = self.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        return output
    
    def multi_scale_deformable_attn_pytorch(self, 
            value: torch.Tensor, value_spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor,
            attention_weights: torch.Tensor) -> torch.Tensor:
        """CPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """
        bs, _, bev_h, bev_w = value.shape

        _, _, _, num_query =\
            sampling_locations.shape

        sampling_value_list = []

        for level, (H_, W_) in enumerate(value_spatial_shapes):

            value_l_ = value.reshape(
                bs * self.num_heads, -1, H_, W_)

            sampling_grid_l_ = sampling_locations.view(-1, self.num_points, 2, num_query).permute(0, 3, 1, 2).contiguous()

            value_l_ = value_l_.split(bs, dim = 0)
            sampling_grid_l_ = sampling_grid_l_.split(bs, dim=0)

            if not self.training:
                sampling_value_l_ = BatchGridSampleFunction.apply(value_l_, sampling_grid_l_, self.grid_sample_mode)
            else:
                sampling_value_l_ = [F.grid_sample(value_, grid_, mode=self.grid_sample_mode, padding_mode='zeros', align_corners=False) for value_, grid_ in zip(value_l_, sampling_grid_l_)]
            
            sampling_value_list.append(sampling_value_l_)

        output = [(sampling_value_list_i * attention_weights_i).sum(-1) for sampling_value_list_i, attention_weights_i in zip(sampling_value_list[0], attention_weights.split(bs, dim=0))]
        output = torch.cat(output, dim=0).view(bs, self.num_heads * self.dim_per_head,
                                                bev_h, bev_w)

        return output


@ATTENTION.register_module()
class MSDeformableAttention3D_SCA_Optimized(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dim_per_head=32,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 use_optimized_linear = True,
                 grid_sample_mode = "nearest",
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        if dim_per_head is None:
            dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.grid_sample_mode = grid_sample_mode

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_points = num_points
        self.use_optimized_linear = use_optimized_linear
        if self.use_optimized_linear:
            self.sampling_offsets = OptimizedLinear(
            embed_dims, num_heads * num_levels * num_points * 2, input_layout = 'nhwc', output_layout = 'nchw')
            self.attention_weights = OptimizedLinear(embed_dims,
                                            num_heads * num_levels * num_points, input_layout = 'nhwc', output_layout = 'nhwc')
            self.value_proj = OptimizedLinear(embed_dims, self.num_heads * self.dim_per_head, input_layout = 'nchw', output_layout = 'nchw')
        else:    
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, self.num_heads * self.dim_per_head)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_optimized_linear:
            constant_init(self.sampling_offsets.layer, 0.)
            self.sampling_offsets.layer.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights.layer, val=0., bias=0.)
            xavier_init(self.value_proj.layer, distribution='uniform', bias=0.)
        else:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def multi_scale_deformable_attn_pytorch(
        self,
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
        """CPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        bs, _, _, _ = value.shape
        
        _, _, _, num_queries =\
            sampling_locations.shape
        
        bev_h, bev_w = int(math.sqrt(num_queries)), int(math.sqrt(num_queries))
        
        sampling_value_list = []
        
        for level, (H_, W_) in enumerate(value_spatial_shapes):

            value_l_ = value.reshape(
                bs * self.num_heads, -1, H_, W_)
            
            sampling_grid_l_ = convert_tensor_format(sampling_locations.reshape(-1, self.num_points, 2, num_queries), 'nchw')

            value_l_ = value_l_.split(bs, dim = 0)
            sampling_grid_l_ = sampling_grid_l_.split(bs, dim=0)

            if not self.training:
                sampling_value_l_ = BatchGridSampleFunction.apply(value_l_, sampling_grid_l_, self.grid_sample_mode)
            else:
                sampling_value_l_ = [F.grid_sample(value_, grid_, mode=self.grid_sample_mode, padding_mode='zeros', align_corners=False) for value_, grid_ in zip(value_l_, sampling_grid_l_)]

            sampling_value_list.append(sampling_value_l_)

        output = [(sampling_value_list_i * attention_weights_i).sum(-1) for sampling_value_list_i, attention_weights_i in zip(sampling_value_list[0], attention_weights.split(bs, dim=0))]
        output = torch.cat(output, dim=0).view(bs, self.num_heads * self.dim_per_head,
                                                bev_h, bev_w)

        output = convert_tensor_format(output, 'nhwc')
        return output

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query1, num_query2, _ = query.shape
        bs, _, num_value1, num_value2 = value.shape
        num_value = num_value1 * num_value2
        num_query = num_query1 * num_query2
        
        if not self.use_optimized_linear:
            value = convert_tensor_format(value, 'nhwc')

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)

        if not self.use_optimized_linear:
            value = convert_tensor_format(value, 'nchw')

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        sampling_offsets = self.sampling_offsets(query)
        if not self.use_optimized_linear:
            sampling_offsets = convert_tensor_format(sampling_offsets, 'nchw')

        sampling_offsets = sampling_offsets.view(
            bs, self.num_heads*self.num_levels*self.num_points,2, num_query)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1).permute(0,2,1,3).reshape(-1,1,num_query,self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            offset_normalizer=1/offset_normalizer[None,None,:,:]

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:,:,:,:]*2 - 1
            
            offset_normalizer = 2*offset_normalizer
            reference_points = reference_points.reshape(bs,-1,1,self.num_points)
            sampling_offsets = sampling_offsets *  offset_normalizer.transpose(3, 2)
            sampling_offsets = sampling_offsets.reshape(bs, -1, self.num_points, num_query)
            sampling_locations = reference_points.permute(0, 2, 3, 1) + sampling_offsets

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = self.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
    
@ATTENTION.register_module()
class CustomMSDeformableAttention_Decoder_Optimized(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 use_optimized_linear = True,
                 grid_sample_mode = "nearest",
                 original_code = False,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.grid_sample_mode = grid_sample_mode

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.dim_per_head = dim_per_head
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.use_optimized_linear = use_optimized_linear
        self.original_code = original_code

        if self.use_optimized_linear:
            self.sampling_offsets = OptimizedLinear(embed_dims, num_heads * num_levels * num_points * 2, input_layout = 'nhwc', output_layout = 'nchw')
            self.attention_weights = OptimizedLinear(embed_dims, num_heads * num_levels * num_points, input_layout = 'nhwc', output_layout = 'nchw')
            self.value_proj = OptimizedLinear(embed_dims, embed_dims, input_layout = 'nhwc', output_layout = 'nchw')
            self.output_proj = OptimizedLinear(embed_dims, embed_dims, input_layout = 'nchw', output_layout = 'nhwc')
        else:
            self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        if self.use_optimized_linear:
            constant_init(self.sampling_offsets.layer, 0.)
            self.sampling_offsets.layer.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights.layer, val=0., bias=0.)
            xavier_init(self.value_proj.layer, distribution='uniform', bias=0.)
            xavier_init(self.output_proj.layer, distribution='uniform', bias=0.)
        else:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def multi_scale_deformable_attn_pytorch(
        self,
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
        """CPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """
        bs,_,bev_h,bev_w = value.shape
        
        if bs > 1:
            bs,num_queries,_,_ = sampling_locations.shape

            sampling_grids = sampling_locations
            sampling_value_list = []
            
            for level, (H_, W_) in enumerate(value_spatial_shapes):

                value_l_ = value.reshape(
                    bs * self.num_heads, self.dim_per_head, H_, W_)
                
                sampling_grid_l_ = sampling_grids.view(bs,-1,self.num_heads,self.num_points,2).transpose(2, 1).reshape(bs*self.num_heads,num_queries,self.num_points,-1)

                value_l_ = value_l_.split(bs, dim = 0)
                sampling_grid_l_ = sampling_grid_l_.split(bs, dim=0)

                sampling_value_l_ = [F.grid_sample(value_, grid_, mode=self.grid_sample_mode, padding_mode='zeros', align_corners=False) for value_, grid_ in zip(value_l_, sampling_grid_l_)]
                sampling_value_list.append(sampling_value_l_)

            output = [(sampling_value_list_i * attention_weights_i).sum(-1) for sampling_value_list_i, attention_weights_i in zip(sampling_value_list[0], attention_weights.split(bs, dim=0))]
            output = torch.cat(output, dim=0).reshape(bs, self.num_heads * self.dim_per_head,
                                                    int(num_queries**0.5), int(num_queries**0.5))
        else:
            _, _, query_len_h, query_len_w=\
                sampling_locations.shape
            
            sampling_grids = sampling_locations
            sampling_value_list = []
            
            for level, (H_, W_) in enumerate(value_spatial_shapes):

                value_l_ = value.reshape(
                    bs * self.num_heads, self.dim_per_head, H_, W_)
                
                sampling_grid_l_ = convert_tensor_format(sampling_grids.view(bs *self.num_heads, self.num_points, 2, -1), "nchw")

                value_l_ = value_l_.split(bs, dim = 0)
                sampling_grid_l_ = sampling_grid_l_.split(bs, dim=0)

                if not self.training:
                    sampling_value_l_ = BatchGridSampleFunction.apply(value_l_, sampling_grid_l_, self.grid_sample_mode)
                else:
                    sampling_value_l_ = [F.grid_sample(value_, grid_, mode=self.grid_sample_mode, padding_mode='zeros', align_corners=False) for value_, grid_ in zip(value_l_, sampling_grid_l_)]
                
                sampling_value_list.append(sampling_value_l_)

            output = [(sampling_value_list_i * attention_weights_i).sum(-1) for sampling_value_list_i, attention_weights_i in zip(sampling_value_list[0], attention_weights.split(bs, dim=0))]

            output = torch.cat(output, dim=0).reshape(bs, self.num_heads * self.dim_per_head,
                                                    query_len_h, query_len_w)
            
        return output

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos

        if self.original_code:
            if not self.batch_first:
                # change to (bs, num_query ,embed_dims)
                query = query.permute(1, 0, 2)
                value = value.permute(1, 0, 2)

        bs, num_query1, num_query1, _ = query.shape
        bs, num_value1, num_value2, _ = value.shape
        num_query = num_query1 * num_query1
        num_value = num_value1 * num_value2

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query)

        if not self.use_optimized_linear:
            sampling_offsets = convert_tensor_format(sampling_offsets, 'nchw')
            attention_weights = convert_tensor_format(attention_weights, 'nchw')
            value = convert_tensor_format(value, 'nchw')
        
        if bs>1:
            sampling_offsets = sampling_offsets.permute(0, 2, 3, 1).reshape(bs,num_query,-1,2)
        else:
            sampling_offsets = sampling_offsets.reshape(
                bs*self.num_heads*self.num_levels*self.num_points, 2, num_query1, num_query1)
            
        attention_weights = attention_weights.reshape(
            bs * self.num_heads, 1, self.num_levels * self.num_points, num_query).permute(0, 1, 3, 2)

        attention_weights = attention_weights.softmax(-1)

        if reference_points.shape[1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            offset_normalizer=1/offset_normalizer[None,None,:,:]
            reference_points = reference_points[:,:,:,:]*2 -1
            offset_normalizer = 2*offset_normalizer
            if bs > 1:
                sampling_locations = (reference_points.permute(0, 2, 3, 1).reshape(bs, num_query, 1, -1) + sampling_offsets *  offset_normalizer)
            else:
                sampling_locations = (reference_points + sampling_offsets *  offset_normalizer.transpose(3, 1))

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        
        output = self.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)

        if self.original_code:
            if not self.batch_first:
                # (num_query, bs ,embed_dims)
                output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
    

==== custom utils ====

import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.autograd import Function

def atan2(y, x):
    """
    Calculates the arctangent of the two variables y and x.
    Similar to calling torch.atan(y / x), but considers the signs of both inputs.
    Returns a new tensor with the same size as input, holding the element-wise atan2 
    of the elements in y over the elements in x.

    Args:
        y (Tensor): The y coordinates.
        x (Tensor): The x coordinates.
    Returns:
        Tensor: The resulting tensor with element-wise atan2 values.
    """

    out = torch.atan(y / (x + 1e-8))
    out += ((y > 0) & (x < 0)) * torch.pi
    out -= ((y < 0) & (x < 0)) * torch.pi
    out *= (1 - ((y > 0) & (x == 0)) * 1.0)
    out += ((y > 0) & (x == 0)) * (torch.pi / 2)
    out *= (1 - ((y < 0) & (x == 0)) * 1.0)
    out += ((y < 0) & (x == 0)) * (-torch.pi / 2)
    return out

def convert_tensor_format(input_tensor, format):
    """
    Converts the format of a tensor from 'nchw' to 'nhwc' or vice versa.

    Args:
        tensor (Tensor): The input tensor to be converted.
        format (str): The desired format of the tensor, either 'nhwc' or 'nchw'.
    Returns:
        Tensor: The converted tensor.

    """

    if format.lower() == 'nhwc':
        return input_tensor.permute(0, 2, 3, 1)
    elif format.lower() == 'nchw':
        return input_tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError("Desired Format must be either 'nhwc' or 'nchw'")

def select_best_indices(bev_h, bev_w):
    """
    Selects the best indices for a given tensor size.
    Calculates the optimal size as 60% of the input size and returns 
    the square root of the product of the optimal height and width.

    Args:
        bev_h (int): The height of the tensor.
        bev_w (int): The width of the tensor.
    Returns:
        float: The square root of the product of the optimal height and width.
    """

    optimal_size_h = int(bev_h * 0.6)
    optimal_size_w = int(bev_w * 0.6)

    return int((optimal_size_h * optimal_size_w)**0.5)**2

def custom_rotate(input_tensor, angle, center=None):
    """
    Rotates an input tensor by a given angle around a specified center point.
    Uses bilinear interpolation for the rotation, resulting in a smoother and more natural-looking image.

    Args:
        input_tensor (Tensor): The input tensor to be rotated.
        angle (float): The angle by which to rotate the tensor, in degrees.
        center (tuple, optional): The center point around which to rotate the tensor. 
                                    If None, the center of the tensor is used.
    Returns:
        Tensor: The rotated tensor.
    """

    # Convert the angle from degrees to radians
    angle = angle * (math.pi / 180.0)

    # Get the size of the input tensor
    batch_size, height, width, num_channels = input_tensor.size()

    # Set the center point
    if center is None:
        center = (width / 2, height / 2)

    # Create a grid of coordinates for the input tensor
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    y, x = y.to(torch.float32), x.to(torch.float32)
    x_c, y_c = x - center[0], y - center[1]

    # Apply the rotation matrix to the coordinates
    new_x = x_c * torch.cos(angle) - y_c * torch.sin(angle) + center[0]
    new_y = x_c * torch.sin(angle) + y_c * torch.cos(angle) + center[1]

    # Stack x and y coordinates
    grid = torch.stack((new_x, new_y), dim=-1)

    # Reshape for grid_sample
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Rescale to [-1, 1]
    grid[..., 0] = grid[..., 0] / (width - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (height - 1) * 2 - 1

    # Permute the input tensor to NCHW format before applying grid_sample
    new_tensor = torch.nn.functional.grid_sample(input_tensor.permute(0, 3, 1, 2), grid.to(input_tensor.device), mode="nearest", align_corners=True)
    # Permute back to NHWC format
    return new_tensor.permute(0, 2, 3, 1)


class OptimizedLinear(nn.Module):
    """
    This class extends the nn.Linear class to create an optimized linear layer.
    It allows for flexible input and output layouts and uses a 
    1x1 Conv2D layer for the linear transformation.
    """

    def __init__(self, in_features, out_features, bias=True, input_layout=None, output_layout=None):
        """
        Initializes the OptimizedLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. 
                                    Default: True.
            input_layout (str, optional): The layout of the input tensors. If 'nhwc', 
                                        the input will be converted to 'nchw'. Default: None.
            output_layout (str, optional): The layout of the output tensors. If 'nhwc' or None, 
                                            the output will be converted to 'nhwc'. Default: None.
        """
        super(OptimizedLinear, self).__init__()
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.layer = nn.Conv2d(in_features, out_features, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        if self.input_layout is None or self.input_layout == 'nhwc':
            x = convert_tensor_format(x, 'nchw')  # Convert input tensor format from 'nhwc' to 'nchw'
        
        x = self.layer(x)  # Apply the 1x1 Conv2D layer
        
        if self.output_layout == 'nhwc' or self.output_layout is None:
            return convert_tensor_format(x, 'nhwc')  # Convert output tensor format to 'nhwc'

        return x

def custom_inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    zero_tensor = torch.zeros_like(x)
    one_tensor = torch.ones_like(x)

    x = torch.max(zero_tensor, x)
    x = torch.min(one_tensor, x)

    x1 = torch.max(torch.full_like(x, eps), x)
    x2 = torch.max(torch.full_like(x, eps), one_tensor - x)
    return torch.log(x1/x2)

class OptimizedFunctionalLinear(nn.Module):
    def __init__(self, input_layout=None, output_layout = None):
        super(OptimizedFunctionalLinear, self).__init__()
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.layer = torch.nn.functional.conv2d

    def forward(self, x, weights, bias):
        """
        Defines the computation performed at every call.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor.
        """
        if self.input_layout is None or self.input_layout == 'nhwc':
            x = convert_tensor_format(x, 'nchw')  # Convert input tensor format from 'nhwc' to 'nchw'
        
        x = self.layer(x, weights.unsqueeze(-1).unsqueeze(-1), bias)  # Apply the 1x1 Conv2D layer
        
        if self.output_layout == 'nhwc' or self.output_layout is None:
            return convert_tensor_format(x, 'nhwc')  # Convert output tensor format to 'nhwc'

        return x

class ScatterND(Function):
    """Custom scatter function to update slots in a canvas using scatter operation
    during export as part of optimization.
    ScatterND autograd usage helps avoiding extra node creations in the onnx.
    Args:
            canvas (torch.Tensor): The canvas tensor.
            indices (list of torch.Tensor): List of index tensors.
            updates (list of torch.Tensor): List of update tensors.
    """
    @staticmethod
    def symbolic(g, canvas, indices, updates):
        return g.op("ScatterND", canvas, indices, updates, reduction_s="add")
    @staticmethod
    def forward(ctx, canvas, indices, updates):
        canvas_ = canvas.clone()
        for i, index in enumerate(indices):
            canvas_[index[:, :, 0], index[:, :, 1]] += updates[i]
        return canvas_

class BatchGridSampleFunction(Function):
    """
    Custom grid_sample function to perform batched grid sampling on input tensors using either bilinear or nearest-neighbor interpolation.
    This custom autograd function can be used during ONNX export to encapsulate
    the grid sampling operation, potentially optimizing the export process.
    Args:
        input list[(torch.Tensor)]: The list of input tensors.
        grid list[(torch.Tensor)]: The list grid tensors.
        mode (str): Interpolation mode ('bilinear' or 'nearest').
    """
    @staticmethod
    def symbolic(g, input, grid, mode):
        # Define the ONNX symbolic representation for the custom operation
        return g.op("GridSample", input, grid, mode_s=mode, padding_mode_s='zeros', align_corners_i=0)

    @staticmethod
    def forward(ctx, input, grid, mode='bilinear'):
        # Perform the grid sampling operation using the specified interpolation mode
        output = [F.grid_sample(value_, grid_, mode=mode, padding_mode='zeros', align_corners=False) for value_, grid_ in zip(input, grid)]
        return output