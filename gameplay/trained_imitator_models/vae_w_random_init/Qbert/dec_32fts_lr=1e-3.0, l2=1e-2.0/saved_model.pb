��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-0-g582c8d236cb8��	
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 ��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
 ��*
dtype0
r
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_namedense_1/bias
k
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes

:��*
dtype0
�
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconv2d_transpose/kernel
�
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_1/kernel
�
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0
�
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_2/kernel
�
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
8
0
1
2
3
4
5
"6
#7
8
0
1
2
3
4
5
"6
#7
 
�
(layer_metrics
	variables

)layers
*non_trainable_variables
+layer_regularization_losses
trainable_variables
,metrics
	regularization_losses
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
-layer_metrics
	variables

.layers
/non_trainable_variables
0layer_regularization_losses
trainable_variables
1metrics
regularization_losses
 
 
 
�
2layer_metrics
	variables

3layers
4non_trainable_variables
5layer_regularization_losses
trainable_variables
6metrics
regularization_losses
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
7layer_metrics
	variables

8layers
9non_trainable_variables
:layer_regularization_losses
trainable_variables
;metrics
regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
<layer_metrics
	variables

=layers
>non_trainable_variables
?layer_regularization_losses
trainable_variables
@metrics
 regularization_losses
ec
VARIABLE_VALUEconv2d_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
Alayer_metrics
$	variables

Blayers
Cnon_trainable_variables
Dlayer_regularization_losses
%trainable_variables
Emetrics
&regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_2Placeholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_1/kerneldense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_736685
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_737097
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_737131�	
�
�
__inference_loss_fn_3_737050H
Dconv2d_transpose_2_kernel_regularizer_square_readvariableop_resource
identity��;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDconv2d_transpose_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentity-conv2d_transpose_2/kernel/Regularizer/mul:z:0<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_decoder_layer_call_fn_736916

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_7365492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_736685
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_7361962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������TT2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�0
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_736237

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_736960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2	
BiasAddZ
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������2
Relu�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
__inference__traced_save_737097
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*s
_input_shapesb
`: :
 ��:��:@@:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
 ��:"

_output_shapes

:��:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::	

_output_shapes
: 
�I
�
C__inference_decoder_layer_call_and_return_conditional_losses_736497
input_2
dense_1_736451
dense_1_736453
conv2d_transpose_736457
conv2d_transpose_736459
conv2d_transpose_1_736462
conv2d_transpose_1_736464
conv2d_transpose_2_736467
conv2d_transpose_2_736469
identity��(conv2d_transpose/StatefulPartitionedCall�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_1/StatefulPartitionedCall�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_2/StatefulPartitionedCall�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_1_736451dense_1_736453*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_7363702!
dense_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_7364002
reshape/PartitionedCall�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_736457conv2d_transpose_736459*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7362372*
(conv2d_transpose/StatefulPartitionedCall�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_736462conv2d_transpose_1_736464*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_7362882,
*conv2d_transpose_1/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_736467conv2d_transpose_2_736469*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_7363392,
*conv2d_transpose_2/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_736451* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_736457*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_736462*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_736467*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
�
3__inference_conv2d_transpose_1_layer_call_fn_736298

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_7362882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_736983

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_2_layer_call_fn_736349

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_7363392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
D
(__inference_reshape_layer_call_fn_736988

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_7364002
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_decoder_layer_call_fn_736568
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_7365492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_736400

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�0
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_736288

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
1__inference_conv2d_transpose_layer_call_fn_736247

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7362372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�I
�
C__inference_decoder_layer_call_and_return_conditional_losses_736619

inputs
dense_1_736573
dense_1_736575
conv2d_transpose_736579
conv2d_transpose_736581
conv2d_transpose_1_736584
conv2d_transpose_1_736586
conv2d_transpose_2_736589
conv2d_transpose_2_736591
identity��(conv2d_transpose/StatefulPartitionedCall�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_1/StatefulPartitionedCall�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_2/StatefulPartitionedCall�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_736573dense_1_736575*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_7363702!
dense_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_7364002
reshape/PartitionedCall�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_736579conv2d_transpose_736581*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7362372*
(conv2d_transpose/StatefulPartitionedCall�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_736584conv2d_transpose_1_736586*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_7362882,
*conv2d_transpose_1/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_736589conv2d_transpose_2_736591*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_7363392,
*conv2d_transpose_2/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_736573* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_736579*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_736584*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_736589*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
Ӥ
�
C__inference_decoder_layer_call_and_return_conditional_losses_736790

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity��'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
dense_1/BiasAddr
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*)
_output_shapes
:�����������2
dense_1/Reluh
reshape/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape�
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack�
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1�
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3�
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack�
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack�
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1�
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose/BiasAdd�
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose/Relu�
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape�
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack�
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1�
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :*2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack�
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack�
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1�
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������** *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������** 2
conv2d_transpose_1/BiasAdd�
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������** 2
conv2d_transpose_1/Relu�
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape�
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack�
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1�
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :T2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :T2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3�
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack�
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack�
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1�
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:���������TT*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT2
conv2d_transpose_2/BiasAdd�
conv2d_transpose_2/SigmoidSigmoid#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������TT2
conv2d_transpose_2/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentityconv2d_transpose_2/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������TT2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_736196
input_22
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resourceE
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource<
8decoder_conv2d_transpose_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_1_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource
identity��/decoder/conv2d_transpose/BiasAdd/ReadVariableOp�8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp�:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�&decoder/dense_1/BiasAdd/ReadVariableOp�%decoder/dense_1/MatMul/ReadVariableOp�
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp�
decoder/dense_1/MatMulMatMulinput_2-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
decoder/dense_1/MatMul�
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOp�
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
decoder/dense_1/BiasAdd�
decoder/dense_1/ReluRelu decoder/dense_1/BiasAdd:output:0*
T0*)
_output_shapes
:�����������2
decoder/dense_1/Relu�
decoder/reshape/ShapeShape"decoder/dense_1/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape�
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack�
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1�
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2�
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice�
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/1�
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2�
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2!
decoder/reshape/Reshape/shape/3�
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape�
decoder/reshape/ReshapeReshape"decoder/dense_1/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
decoder/reshape/Reshape�
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/Shape�
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder/conv2d_transpose/strided_slice/stack�
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_1�
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_2�
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/conv2d_transpose/strided_slice�
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/1�
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/2�
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 decoder/conv2d_transpose/stack/3�
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/stack�
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose/strided_slice_1/stack�
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_1�
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_2�
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose/strided_slice_1�
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp�
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2+
)decoder/conv2d_transpose/conv2d_transpose�
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp�
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2"
 decoder/conv2d_transpose/BiasAdd�
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
decoder/conv2d_transpose/Relu�
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/Shape�
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_1/strided_slice/stack�
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_1�
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_2�
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_1/strided_slice�
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :*2$
"decoder/conv2d_transpose_1/stack/1�
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*2$
"decoder/conv2d_transpose_1/stack/2�
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_1/stack/3�
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/stack�
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_1/strided_slice_1/stack�
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_1�
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_2�
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_1/strided_slice_1�
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������** *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_1/conv2d_transpose�
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������** 2$
"decoder/conv2d_transpose_1/BiasAdd�
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������** 2!
decoder/conv2d_transpose_1/Relu�
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/Shape�
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_2/strided_slice/stack�
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_1�
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_2�
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_2/strided_slice�
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :T2$
"decoder/conv2d_transpose_2/stack/1�
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :T2$
"decoder/conv2d_transpose_2/stack/2�
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/3�
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/stack�
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_2/strided_slice_1/stack�
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_1�
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_2�
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_2/strided_slice_1�
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:���������TT*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_2/conv2d_transpose�
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp�
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT2$
"decoder/conv2d_transpose_2/BiasAdd�
"decoder/conv2d_transpose_2/SigmoidSigmoid+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������TT2$
"decoder/conv2d_transpose_2/Sigmoid�
IdentityIdentity&decoder/conv2d_transpose_2/Sigmoid:y:00^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:���������TT2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�&
�
"__inference__traced_restore_737131
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias.
*assignvariableop_2_conv2d_transpose_kernel,
(assignvariableop_3_conv2d_transpose_bias0
,assignvariableop_4_conv2d_transpose_1_kernel.
*assignvariableop_5_conv2d_transpose_1_bias0
,assignvariableop_6_conv2d_transpose_2_kernel.
*assignvariableop_7_conv2d_transpose_2_bias

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv2d_transpose_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv2d_transpose_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_decoder_layer_call_fn_736937

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_7366192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
Ӥ
�
C__inference_decoder_layer_call_and_return_conditional_losses_736895

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity��'conv2d_transpose/BiasAdd/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�)conv2d_transpose_1/BiasAdd/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�)conv2d_transpose_2/BiasAdd/ReadVariableOp�2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
dense_1/BiasAddr
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*)
_output_shapes
:�����������2
dense_1/Reluh
reshape/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapedense_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape�
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack�
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1�
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3�
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack�
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack�
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1�
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose�
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp�
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose/BiasAdd�
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_transpose/Relu�
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape�
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack�
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1�
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :*2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :*2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3�
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack�
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack�
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1�
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������** *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose�
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp�
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������** 2
conv2d_transpose_1/BiasAdd�
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������** 2
conv2d_transpose_1/Relu�
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape�
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack�
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1�
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2�
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :T2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :T2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3�
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack�
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack�
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1�
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2�
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1�
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp�
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:���������TT*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose�
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp�
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT2
conv2d_transpose_2/BiasAdd�
conv2d_transpose_2/SigmoidSigmoid#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������TT2
conv2d_transpose_2/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentityconv2d_transpose_2/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������TT2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_737017=
9dense_1_kernel_regularizer_square_readvariableop_resource
identity��0dense_1/kernel/Regularizer/Square/ReadVariableOp�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
�I
�
C__inference_decoder_layer_call_and_return_conditional_losses_736549

inputs
dense_1_736503
dense_1_736505
conv2d_transpose_736509
conv2d_transpose_736511
conv2d_transpose_1_736514
conv2d_transpose_1_736516
conv2d_transpose_2_736519
conv2d_transpose_2_736521
identity��(conv2d_transpose/StatefulPartitionedCall�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_1/StatefulPartitionedCall�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_2/StatefulPartitionedCall�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_736503dense_1_736505*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_7363702!
dense_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_7364002
reshape/PartitionedCall�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_736509conv2d_transpose_736511*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7362372*
(conv2d_transpose/StatefulPartitionedCall�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_736514conv2d_transpose_1_736516*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_7362882,
*conv2d_transpose_1/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_736519conv2d_transpose_2_736521*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_7363392,
*conv2d_transpose_2/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_736503* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_736509*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_736514*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_736519*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�0
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_736339

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2	
Sigmoid�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�I
�
C__inference_decoder_layer_call_and_return_conditional_losses_736448
input_2
dense_1_736381
dense_1_736383
conv2d_transpose_736408
conv2d_transpose_736410
conv2d_transpose_1_736413
conv2d_transpose_1_736415
conv2d_transpose_2_736418
conv2d_transpose_2_736420
identity��(conv2d_transpose/StatefulPartitionedCall�9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_1/StatefulPartitionedCall�;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�*conv2d_transpose_2/StatefulPartitionedCall�;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_1_736381dense_1_736383*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_7363702!
dense_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_7364002
reshape/PartitionedCall�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_736408conv2d_transpose_736410*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_7362372*
(conv2d_transpose/StatefulPartitionedCall�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_736413conv2d_transpose_1_736415*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_7362882,
*conv2d_transpose_1/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_736418conv2d_transpose_2_736420*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_7363392,
*conv2d_transpose_2/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_736381* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_736408*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_736413*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_736418*&
_output_shapes
: *
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_2/kernel/Regularizer/Square�
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_2/kernel/Regularizer/Const�
)conv2d_transpose_2/kernel/Regularizer/SumSum0conv2d_transpose_2/kernel/Regularizer/Square:y:04conv2d_transpose_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum�
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x�
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul�
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
�
__inference_loss_fn_2_737039H
Dconv2d_transpose_1_kernel_regularizer_square_readvariableop_resource
identity��;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDconv2d_transpose_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp�
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2.
,conv2d_transpose_1/kernel/Regularizer/Square�
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose_1/kernel/Regularizer/Const�
)conv2d_transpose_1/kernel/Regularizer/SumSum0conv2d_transpose_1/kernel/Regularizer/Square:y:04conv2d_transpose_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum�
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x�
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul�
IdentityIdentity-conv2d_transpose_1/kernel/Regularizer/mul:z:0<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_decoder_layer_call_fn_736638
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_7366192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:��������� ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
}
(__inference_dense_1_layer_call_fn_736969

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_7363702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_737028F
Bconv2d_transpose_kernel_regularizer_square_readvariableop_resource
identity��9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp�
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2,
*conv2d_transpose/kernel/Regularizer/Square�
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)conv2d_transpose/kernel/Regularizer/Const�
'conv2d_transpose/kernel/Regularizer/SumSum.conv2d_transpose/kernel/Regularizer/Square:y:02conv2d_transpose/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum�
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2+
)conv2d_transpose/kernel/Regularizer/mul/x�
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul�
IdentityIdentity+conv2d_transpose/kernel/Regularizer/mul:z:0:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_736370

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������2	
BiasAddZ
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������2
Relu�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ��*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
 ��2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_20
serving_default_input_2:0��������� N
conv2d_transpose_28
StatefulPartitionedCall:0���������TTtensorflow/serving/predict:��
�A
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
*F&call_and_return_all_conditional_losses
G__call__
H_default_save_signature"�>
_tf_keras_network�={"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 28224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [21, 21, 64]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 28224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [21, 21, 64]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_2", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 28224, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [21, 21, 64]}}}
�


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 64]}}
�


kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
*O&call_and_return_all_conditional_losses
P__call__"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 64]}}
�


"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"�	
_tf_keras_layer�	{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 32]}}
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
<
S0
T1
U2
V3"
trackable_list_wrapper
�
(layer_metrics
	variables

)layers
*non_trainable_variables
+layer_regularization_losses
trainable_variables
,metrics
	regularization_losses
G__call__
H_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
": 
 ��2dense_1/kernel
:��2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
S0"
trackable_list_wrapper
�
-layer_metrics
	variables

.layers
/non_trainable_variables
0layer_regularization_losses
trainable_variables
1metrics
regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
2layer_metrics
	variables

3layers
4non_trainable_variables
5layer_regularization_losses
trainable_variables
6metrics
regularization_losses
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
1:/@@2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
T0"
trackable_list_wrapper
�
7layer_metrics
	variables

8layers
9non_trainable_variables
:layer_regularization_losses
trainable_variables
;metrics
regularization_losses
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
U0"
trackable_list_wrapper
�
<layer_metrics
	variables

=layers
>non_trainable_variables
?layer_regularization_losses
trainable_variables
@metrics
 regularization_losses
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
�
Alayer_metrics
$	variables

Blayers
Cnon_trainable_variables
Dlayer_regularization_losses
%trainable_variables
Emetrics
&regularization_losses
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
U0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
C__inference_decoder_layer_call_and_return_conditional_losses_736895
C__inference_decoder_layer_call_and_return_conditional_losses_736448
C__inference_decoder_layer_call_and_return_conditional_losses_736790
C__inference_decoder_layer_call_and_return_conditional_losses_736497�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_decoder_layer_call_fn_736568
(__inference_decoder_layer_call_fn_736916
(__inference_decoder_layer_call_fn_736638
(__inference_decoder_layer_call_fn_736937�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_736196�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_2��������� 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_736960�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_736969�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_reshape_layer_call_and_return_conditional_losses_736983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_reshape_layer_call_fn_736988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_736237�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
1__inference_conv2d_transpose_layer_call_fn_736247�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_736288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
3__inference_conv2d_transpose_1_layer_call_fn_736298�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_736339�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
3__inference_conv2d_transpose_2_layer_call_fn_736349�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
__inference_loss_fn_0_737017�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_737028�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_737039�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_737050�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
$__inference_signature_wrapper_736685input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_736196�"#0�-
&�#
!�
input_2��������� 
� "O�L
J
conv2d_transpose_24�1
conv2d_transpose_2���������TT�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_736288�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
3__inference_conv2d_transpose_1_layer_call_fn_736298�I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_736339�"#I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
3__inference_conv2d_transpose_2_layer_call_fn_736349�"#I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_736237�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_conv2d_transpose_layer_call_fn_736247�I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
C__inference_decoder_layer_call_and_return_conditional_losses_736448�"#8�5
.�+
!�
input_2��������� 
p

 
� "?�<
5�2
0+���������������������������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_736497�"#8�5
.�+
!�
input_2��������� 
p 

 
� "?�<
5�2
0+���������������������������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_736790r"#7�4
-�*
 �
inputs��������� 
p

 
� "-�*
#� 
0���������TT
� �
C__inference_decoder_layer_call_and_return_conditional_losses_736895r"#7�4
-�*
 �
inputs��������� 
p 

 
� "-�*
#� 
0���������TT
� �
(__inference_decoder_layer_call_fn_736568x"#8�5
.�+
!�
input_2��������� 
p

 
� "2�/+����������������������������
(__inference_decoder_layer_call_fn_736638x"#8�5
.�+
!�
input_2��������� 
p 

 
� "2�/+����������������������������
(__inference_decoder_layer_call_fn_736916w"#7�4
-�*
 �
inputs��������� 
p

 
� "2�/+����������������������������
(__inference_decoder_layer_call_fn_736937w"#7�4
-�*
 �
inputs��������� 
p 

 
� "2�/+����������������������������
C__inference_dense_1_layer_call_and_return_conditional_losses_736960^/�,
%�"
 �
inputs��������� 
� "'�$
�
0�����������
� }
(__inference_dense_1_layer_call_fn_736969Q/�,
%�"
 �
inputs��������� 
� "������������;
__inference_loss_fn_0_737017�

� 
� "� ;
__inference_loss_fn_1_737028�

� 
� "� ;
__inference_loss_fn_2_737039�

� 
� "� ;
__inference_loss_fn_3_737050"�

� 
� "� �
C__inference_reshape_layer_call_and_return_conditional_losses_736983b1�.
'�$
"�
inputs�����������
� "-�*
#� 
0���������@
� �
(__inference_reshape_layer_call_fn_736988U1�.
'�$
"�
inputs�����������
� " ����������@�
$__inference_signature_wrapper_736685�"#;�8
� 
1�.
,
input_2!�
input_2��������� "O�L
J
conv2d_transpose_24�1
conv2d_transpose_2���������TT