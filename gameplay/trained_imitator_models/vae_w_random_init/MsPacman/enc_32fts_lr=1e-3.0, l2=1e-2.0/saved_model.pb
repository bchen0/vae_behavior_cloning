Ѕ
шО
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

Conv2D

input"T
filter"T
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
,
Exp
x"T
y"T"
Ttype:

2
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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-0-g582c8d236cb8кы
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Рм *
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
Рм *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:  *
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
: *
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:  *
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
: *
dtype0

NoOpNoOp
#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н"
valueГ"BА" BЉ"
ѓ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
R
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
 
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
­
;layer_regularization_losses
<non_trainable_variables

regularization_losses
	variables

=layers
>layer_metrics
?metrics
trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
@layer_regularization_losses
Anon_trainable_variables
regularization_losses
	variables

Blayers
Clayer_metrics
Dmetrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Elayer_regularization_losses
Fnon_trainable_variables
regularization_losses
	variables

Glayers
Hlayer_metrics
Imetrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Jlayer_regularization_losses
Knon_trainable_variables
regularization_losses
	variables

Llayers
Mlayer_metrics
Nmetrics
trainable_variables
 
 
 
­
Olayer_regularization_losses
Pnon_trainable_variables
!regularization_losses
"	variables

Qlayers
Rlayer_metrics
Smetrics
#trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
­
Tlayer_regularization_losses
Unon_trainable_variables
'regularization_losses
(	variables

Vlayers
Wlayer_metrics
Xmetrics
)trainable_variables
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
­
Ylayer_regularization_losses
Znon_trainable_variables
-regularization_losses
.	variables

[layers
\layer_metrics
]metrics
/trainable_variables
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
­
^layer_regularization_losses
_non_trainable_variables
3regularization_losses
4	variables

`layers
alayer_metrics
bmetrics
5trainable_variables
 
 
 
­
clayer_regularization_losses
dnon_trainable_variables
7regularization_losses
8	variables

elayers
flayer_metrics
gmetrics
9trainable_variables
 
 
?
0
1
2
3
4
5
6
7
	8
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

serving_default_input_1Placeholder*/
_output_shapes
:џџџџџџџџџTT*
dtype0*$
shape:џџџџџџџџџTT
Ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1112621
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1113257
з
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1113303Ѓ
с

+__inference_z_log_var_layer_call_fn_1113098

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_11121712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Д

B__inference_dense_layer_call_and_return_conditional_losses_1113027

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ.dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ReluС
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџРм::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџРм
 
_user_specified_nameinputs
ф

C__inference_z_mean_layer_call_and_return_conditional_losses_1112139

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddС
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulЧ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
И

C__inference_conv2d_layer_call_and_return_conditional_losses_1111994

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ/conv2d/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
ReluЩ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ** 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџTT::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
ъ
Ж
)__inference_encoder_layer_call_fn_1112443
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2ЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_11124122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1
І
E
)__inference_flatten_layer_call_fn_1113004

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџРм* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_11120822
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ў

*__inference_conv2d_2_layer_call_fn_1112993

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_11120602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
іm
Ё
D__inference_encoder_layer_call_and_return_conditional_losses_1112261
input_1
conv2d_1112005
conv2d_1112007
conv2d_1_1112038
conv2d_1_1112040
conv2d_2_1112071
conv2d_2_1112073
dense_1112118
dense_1112120
z_mean_1112150
z_mean_1112152
z_log_var_1112182
z_log_var_1112184
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_1/StatefulPartitionedCallЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_2/StatefulPartitionedCallЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/StatefulPartitionedCallЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/StatefulPartitionedCallЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1112005conv2d_1112007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ** *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_11119942 
conv2d/StatefulPartitionedCallР
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1112038conv2d_1_1112040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_11120272"
 conv2d_1/StatefulPartitionedCallТ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1112071conv2d_2_1112073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_11120602"
 conv2d_2/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџРм* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_11120822
flatten/PartitionedCallЂ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1112118dense_1112120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11121072
dense/StatefulPartitionedCall­
z_mean/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_mean_1112150z_mean_1112152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_11121392 
z_mean/StatefulPartitionedCallМ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_log_var_1112182z_log_var_1112184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_11121712#
!z_log_var/StatefulPartitionedCallЛ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_11122132"
 sampling/StatefulPartitionedCallЙ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1112005*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulП
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1112038*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulП
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1112071*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulА
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1112118* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulБ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_mean_1112150*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulК
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_log_var_1112182*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mul
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЃ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1Ђ

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1
Ѕ
s
*__inference_sampling_layer_call_fn_1113130
inputs_0
inputs_1
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_11122132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
О
`
D__inference_flatten_layer_call_and_return_conditional_losses_1112082

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@n  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ф

E__inference_conv2d_2_layer_call_and_return_conditional_losses_1112984

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ReluЭ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulг
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
І

F__inference_z_log_var_layer_call_and_return_conditional_losses_1112171

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddЧ
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mulЪ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
м
|
'__inference_dense_layer_call_fn_1113036

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11121072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџРм::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџРм
 
_user_specified_nameinputs
Х
Є
__inference_loss_fn_2_1113163>
:conv2d_2_kernel_regularizer_square_readvariableop_resource
identityЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpщ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul
IdentityIdentity#conv2d_2/kernel/Regularizer/mul:z:02^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
ф

E__inference_conv2d_1_layer_call_and_return_conditional_losses_1112952

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ReluЭ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulг
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ** ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ** 
 
_user_specified_nameinputs
Х
Є
__inference_loss_fn_1_1113152>
:conv2d_1_kernel_regularizer_square_readvariableop_resource
identityЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpщ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul
IdentityIdentity#conv2d_1/kernel/Regularizer/mul:z:02^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
ѓm
 
D__inference_encoder_layer_call_and_return_conditional_losses_1112412

inputs
conv2d_1112341
conv2d_1112343
conv2d_1_1112346
conv2d_1_1112348
conv2d_2_1112351
conv2d_2_1112353
dense_1112357
dense_1112359
z_mean_1112362
z_mean_1112364
z_log_var_1112367
z_log_var_1112369
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_1/StatefulPartitionedCallЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_2/StatefulPartitionedCallЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/StatefulPartitionedCallЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/StatefulPartitionedCallЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1112341conv2d_1112343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ** *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_11119942 
conv2d/StatefulPartitionedCallР
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1112346conv2d_1_1112348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_11120272"
 conv2d_1/StatefulPartitionedCallТ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1112351conv2d_2_1112353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_11120602"
 conv2d_2/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџРм* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_11120822
flatten/PartitionedCallЂ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1112357dense_1112359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11121072
dense/StatefulPartitionedCall­
z_mean/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_mean_1112362z_mean_1112364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_11121392 
z_mean/StatefulPartitionedCallМ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_log_var_1112367z_log_var_1112369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_11121712#
!z_log_var/StatefulPartitionedCallЛ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_11122132"
 sampling/StatefulPartitionedCallЙ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1112341*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulП
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1112346*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulП
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1112351*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulА
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1112357* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulБ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_mean_1112362*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulК
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_log_var_1112367*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mul
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЃ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1Ђ

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
ф

E__inference_conv2d_2_layer_call_and_return_conditional_losses_1112060

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ReluЭ
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulг
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О
`
D__inference_flatten_layer_call_and_return_conditional_losses_1112999

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@n  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ф

E__inference_conv2d_1_layer_call_and_return_conditional_losses_1112027

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
ReluЭ
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulг
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ** ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ** 
 
_user_specified_nameinputs
Ф
В
%__inference_signature_wrapper_1112621
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2ЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_11119732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1
З
б

D__inference_encoder_layer_call_and_return_conditional_losses_1112831

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOpЂ/z_mean/kernel/Regularizer/Square/ReadVariableOpЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** *
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
conv2d/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpб
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_1/ReluА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpг
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@n  2
flatten/Const
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

dense/ReluЂ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMuldense/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_mean/MatMulЁ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_mean/BiasAddЋ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
z_log_var/MatMul/ReadVariableOpЃ
z_log_var/MatMulMatMuldense/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_log_var/MatMulЊ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 z_log_var/BiasAdd/ReadVariableOpЉ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2Є
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ж
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddev
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2ў2-
+sampling/random_normal/RandomStandardNormalи
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normal/mulИ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/addа
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulж
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulж
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЧ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulШ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulб
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mulЃ
IdentityIdentityz_mean/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЊ

Identity_1Identityz_log_var/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1 

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs

 
__inference_loss_fn_0_1113141<
8conv2d_kernel_regularizer_square_readvariableop_resource
identityЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpу
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul
IdentityIdentity!conv2d/kernel/Regularizer/mul:z:00^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp
Х
І
__inference_loss_fn_5_1113196?
;z_log_var_kernel_regularizer_square_readvariableop_resource
identityЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpф
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;z_log_var_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mul
IdentityIdentity$z_log_var/kernel/Regularizer/mul:z:03^z_log_var/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp
ъ
Ж
)__inference_encoder_layer_call_fn_1112550
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2ЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_11125192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1
ч
Е
)__inference_encoder_layer_call_fn_1112864

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_11124122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
З
б

D__inference_encoder_layer_call_and_return_conditional_losses_1112726

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOpЂ/z_mean/kernel/Regularizer/Square/ReadVariableOpЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** *
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
conv2d/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOpб
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_1/ReluА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpг
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@n  2
flatten/Const
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

dense/ReluЂ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMuldense/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_mean/MatMulЁ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_mean/BiasAddЋ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02!
z_log_var/MatMul/ReadVariableOpЃ
z_log_var/MatMulMatMuldense/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_log_var/MatMulЊ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 z_log_var/BiasAdd/ReadVariableOpЉ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2Є
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ж
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddev
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2Шж2-
+sampling/random_normal/RandomStandardNormalи
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normal/mulИ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
sampling/addа
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulж
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulж
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulЧ
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulШ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulб
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mulЃ
IdentityIdentityz_mean/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЊ

Identity_1Identityz_log_var/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1 

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
ў

*__inference_conv2d_1_layer_call_fn_1112961

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_11120272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ** ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ** 
 
_user_specified_nameinputs
§
 
__inference_loss_fn_4_1113185<
8z_mean_kernel_regularizer_square_readvariableop_resource
identityЂ/z_mean/kernel/Regularizer/Square/ReadVariableOpл
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8z_mean_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mul
IdentityIdentity!z_mean/kernel/Regularizer/mul:z:00^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp
ј$

 __inference__traced_save_1113257
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЕ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ч
valueНBКB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesО
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*
_input_shapes
~: : : : @:@:@@:@:
Рм : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
Рм : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: 
І

F__inference_z_log_var_layer_call_and_return_conditional_losses_1113089

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddЧ
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mulЪ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^z_log_var/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
њ
}
(__inference_conv2d_layer_call_fn_1112929

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ** *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_11119942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ** 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџTT::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
ф

C__inference_z_mean_layer_call_and_return_conditional_losses_1113058

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddС
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulЧ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
к
}
(__inference_z_mean_layer_call_fn_1113067

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_11121392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ѓm
 
D__inference_encoder_layer_call_and_return_conditional_losses_1112519

inputs
conv2d_1112448
conv2d_1112450
conv2d_1_1112453
conv2d_1_1112455
conv2d_2_1112458
conv2d_2_1112460
dense_1112464
dense_1112466
z_mean_1112469
z_mean_1112471
z_log_var_1112474
z_log_var_1112476
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_1/StatefulPartitionedCallЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_2/StatefulPartitionedCallЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/StatefulPartitionedCallЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/StatefulPartitionedCallЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1112448conv2d_1112450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ** *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_11119942 
conv2d/StatefulPartitionedCallР
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1112453conv2d_1_1112455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_11120272"
 conv2d_1/StatefulPartitionedCallТ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1112458conv2d_2_1112460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_11120602"
 conv2d_2/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџРм* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_11120822
flatten/PartitionedCallЂ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1112464dense_1112466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11121072
dense/StatefulPartitionedCall­
z_mean/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_mean_1112469z_mean_1112471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_11121392 
z_mean/StatefulPartitionedCallМ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_log_var_1112474z_log_var_1112476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_11121712#
!z_log_var/StatefulPartitionedCallЛ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_11122132"
 sampling/StatefulPartitionedCallЙ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1112448*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulП
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1112453*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulП
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1112458*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulА
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1112464* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulБ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_mean_1112469*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulК
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_log_var_1112474*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mul
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЃ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1Ђ

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs

t
E__inference_sampling_layer_call_and_return_conditional_losses_1113124
inputs_0
inputs_1
identityF
ShapeShapeinputs_0*
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
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevх
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2Шсн2$
"random_normal/RandomStandardNormalД
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ :Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
Нl
О	
"__inference__wrapped_model_1111973
input_11
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource3
/encoder_conv2d_1_conv2d_readvariableop_resource4
0encoder_conv2d_1_biasadd_readvariableop_resource3
/encoder_conv2d_2_conv2d_readvariableop_resource4
0encoder_conv2d_2_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђ%encoder/conv2d/BiasAdd/ReadVariableOpЂ$encoder/conv2d/Conv2D/ReadVariableOpЂ'encoder/conv2d_1/BiasAdd/ReadVariableOpЂ&encoder/conv2d_1/Conv2D/ReadVariableOpЂ'encoder/conv2d_2/BiasAdd/ReadVariableOpЂ&encoder/conv2d_2/Conv2D/ReadVariableOpЂ$encoder/dense/BiasAdd/ReadVariableOpЂ#encoder/dense/MatMul/ReadVariableOpЂ(encoder/z_log_var/BiasAdd/ReadVariableOpЂ'encoder/z_log_var/MatMul/ReadVariableOpЂ%encoder/z_mean/BiasAdd/ReadVariableOpЂ$encoder/z_mean/MatMul/ReadVariableOpТ
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOpб
encoder/conv2d/Conv2DConv2Dinput_1,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** *
paddingSAME*
strides
2
encoder/conv2d/Conv2DЙ
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOpФ
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
encoder/conv2d/BiasAdd
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
encoder/conv2d/ReluШ
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOpё
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
encoder/conv2d_1/Conv2DП
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOpЬ
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
encoder/conv2d_1/BiasAdd
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
encoder/conv2d_1/ReluШ
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOpѓ
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
encoder/conv2d_2/Conv2DП
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOpЬ
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
encoder/conv2d_2/BiasAdd
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
encoder/conv2d_2/Relu
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@n  2
encoder/flatten/ConstЖ
encoder/flatten/ReshapeReshape#encoder/conv2d_2/Relu:activations:0encoder/flatten/Const:output:0*
T0*)
_output_shapes
:џџџџџџџџџРм2
encoder/flatten/ReshapeЙ
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype02%
#encoder/dense/MatMul/ReadVariableOpЗ
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/dense/MatMulЖ
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$encoder/dense/BiasAdd/ReadVariableOpЙ
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/dense/BiasAdd
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/dense/ReluК
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$encoder/z_mean/MatMul/ReadVariableOpК
encoder/z_mean/MatMulMatMul encoder/dense/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/z_mean/MatMulЙ
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder/z_mean/BiasAdd/ReadVariableOpН
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/z_mean/BiasAddУ
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02)
'encoder/z_log_var/MatMul/ReadVariableOpУ
encoder/z_log_var/MatMulMatMul encoder/dense/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/z_log_var/MatMulТ
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(encoder/z_log_var/BiasAdd/ReadVariableOpЩ
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/z_log_var/BiasAdd
encoder/sampling/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/Shape
$encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$encoder/sampling/strided_slice/stack
&encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice/stack_1
&encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice/stack_2Ш
encoder/sampling/strided_sliceStridedSliceencoder/sampling/Shape:output:0-encoder/sampling/strided_slice/stack:output:0/encoder/sampling/strided_slice/stack_1:output:0/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
encoder/sampling/strided_slice
encoder/sampling/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/Shape_1
&encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice_1/stack
(encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling/strided_slice_1/stack_1
(encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling/strided_slice_1/stack_2д
 encoder/sampling/strided_slice_1StridedSlice!encoder/sampling/Shape_1:output:0/encoder/sampling/strided_slice_1/stack:output:01encoder/sampling/strided_slice_1/stack_1:output:01encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 encoder/sampling/strided_slice_1ж
$encoder/sampling/random_normal/shapePack'encoder/sampling/strided_slice:output:0)encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2&
$encoder/sampling/random_normal/shape
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder/sampling/random_normal/mean
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%encoder/sampling/random_normal/stddev
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2даF25
3encoder/sampling/random_normal/RandomStandardNormalј
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"encoder/sampling/random_normal/mulи
encoder/sampling/random_normalAdd&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2 
encoder/sampling/random_normalu
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling/mul/xЊ
encoder/sampling/mulMulencoder/sampling/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/sampling/mul
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/sampling/ExpЇ
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/sampling/mul_1Є
encoder/sampling/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder/sampling/addв
IdentityIdentityencoder/sampling/add:z:0&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityр

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1н

Identity_2Identityencoder/z_mean/BiasAdd:output:0&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1

r
E__inference_sampling_layer_call_and_return_conditional_losses_1112213

inputs
inputs_1
identityD
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
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevх
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2љр2$
"random_normal/RandomStandardNormalД
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
5

#__inference__traced_restore_1113303
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias$
 assignvariableop_8_z_mean_kernel"
assignvariableop_9_z_mean_bias(
$assignvariableop_10_z_log_var_kernel&
"assignvariableop_11_z_log_var_bias
identity_13ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ч
valueНBКB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Є
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѕ
AssignVariableOp_8AssignVariableOp assignvariableop_8_z_mean_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOpassignvariableop_9_z_mean_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ќ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_z_log_var_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Њ
AssignVariableOp_11AssignVariableOp"assignvariableop_11_z_log_var_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12й
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
іm
Ё
D__inference_encoder_layer_call_and_return_conditional_losses_1112335
input_1
conv2d_1112264
conv2d_1112266
conv2d_1_1112269
conv2d_1_1112271
conv2d_2_1112274
conv2d_2_1112276
dense_1112280
dense_1112282
z_mean_1112285
z_mean_1112287
z_log_var_1112290
z_log_var_1112292
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ/conv2d/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_1/StatefulPartitionedCallЂ1conv2d_1/kernel/Regularizer/Square/ReadVariableOpЂ conv2d_2/StatefulPartitionedCallЂ1conv2d_2/kernel/Regularizer/Square/ReadVariableOpЂdense/StatefulPartitionedCallЂ.dense/kernel/Regularizer/Square/ReadVariableOpЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂ2z_log_var/kernel/Regularizer/Square/ReadVariableOpЂz_mean/StatefulPartitionedCallЂ/z_mean/kernel/Regularizer/Square/ReadVariableOp
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1112264conv2d_1112266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ** *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_11119942 
conv2d/StatefulPartitionedCallР
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1112269conv2d_1_1112271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_11120272"
 conv2d_1/StatefulPartitionedCallТ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1112274conv2d_2_1112276*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_11120602"
 conv2d_2/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџРм* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_11120822
flatten/PartitionedCallЂ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1112280dense_1112282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_11121072
dense/StatefulPartitionedCall­
z_mean/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_mean_1112285z_mean_1112287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_11121392 
z_mean/StatefulPartitionedCallМ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_log_var_1112290z_log_var_1112292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_11121712#
!z_log_var/StatefulPartitionedCallЛ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sampling_layer_call_and_return_conditional_losses_11122132"
 sampling/StatefulPartitionedCallЙ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1112264*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulП
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_1112269*&
_output_shapes
: @*
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_1/kernel/Regularizer/Square
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_1/kernel/Regularizer/ConstО
conv2d_1/kernel/Regularizer/SumSum&conv2d_1/kernel/Regularizer/Square:y:0*conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_1/kernel/Regularizer/mul/xР
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mulП
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_1112274*&
_output_shapes
:@@*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpО
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2$
"conv2d_2/kernel/Regularizer/Square
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d_2/kernel/Regularizer/ConstО
conv2d_2/kernel/Regularizer/SumSum&conv2d_2/kernel/Regularizer/Square:y:0*conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2#
!conv2d_2/kernel/Regularizer/mul/xР
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mulА
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1112280* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulБ
/z_mean/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_mean_1112285*
_output_shapes

:  *
dtype021
/z_mean/kernel/Regularizer/Square/ReadVariableOpА
 z_mean/kernel/Regularizer/SquareSquare7z_mean/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2"
 z_mean/kernel/Regularizer/Square
z_mean/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
z_mean/kernel/Regularizer/ConstЖ
z_mean/kernel/Regularizer/SumSum$z_mean/kernel/Regularizer/Square:y:0(z_mean/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/Sum
z_mean/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
z_mean/kernel/Regularizer/mul/xИ
z_mean/kernel/Regularizer/mulMul(z_mean/kernel/Regularizer/mul/x:output:0&z_mean/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
z_mean/kernel/Regularizer/mulК
2z_log_var/kernel/Regularizer/Square/ReadVariableOpReadVariableOpz_log_var_1112290*
_output_shapes

:  *
dtype024
2z_log_var/kernel/Regularizer/Square/ReadVariableOpЙ
#z_log_var/kernel/Regularizer/SquareSquare:z_log_var/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:  2%
#z_log_var/kernel/Regularizer/Square
"z_log_var/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"z_log_var/kernel/Regularizer/ConstТ
 z_log_var/kernel/Regularizer/SumSum'z_log_var/kernel/Regularizer/Square:y:0+z_log_var/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/Sum
"z_log_var/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2$
"z_log_var/kernel/Regularizer/mul/xФ
 z_log_var/kernel/Regularizer/mulMul+z_log_var/kernel/Regularizer/mul/x:output:0)z_log_var/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 z_log_var/kernel/Regularizer/mul
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЃ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1Ђ

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall3^z_log_var/kernel/Regularizer/Square/ReadVariableOp^z_mean/StatefulPartitionedCall0^z_mean/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2h
2z_log_var/kernel/Regularizer/Square/ReadVariableOp2z_log_var/kernel/Regularizer/Square/ReadVariableOp2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall2b
/z_mean/kernel/Regularizer/Square/ReadVariableOp/z_mean/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџTT
!
_user_specified_name	input_1
щ

__inference_loss_fn_3_1113174;
7dense_kernel_regularizer_square_readvariableop_resource
identityЂ.dense/kernel/Regularizer/Square/ReadVariableOpк
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul
IdentityIdentity dense/kernel/Regularizer/mul:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
И

C__inference_conv2d_layer_call_and_return_conditional_losses_1112920

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpЂ/conv2d/kernel/Regularizer/Square/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ** 2
ReluЩ
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOpИ
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2"
 conv2d/kernel/Regularizer/Square
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2!
conv2d/kernel/Regularizer/ConstЖ
conv2d/kernel/Regularizer/SumSum$conv2d/kernel/Regularizer/Square:y:0(conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2!
conv2d/kernel/Regularizer/mul/xИ
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mulб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ** 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџTT::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs
Д

B__inference_dense_layer_call_and_return_conditional_losses_1112107

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ.dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
ReluС
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Рм *
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpЏ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Рм 2!
dense/kernel/Regularizer/Square
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/ConstВ
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2 
dense/kernel/Regularizer/mul/xД
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџРм::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџРм
 
_user_specified_nameinputs
ч
Е
)__inference_encoder_layer_call_fn_1112897

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1

identity_2ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_11125192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*^
_input_shapesM
K:џџџџџџџџџTT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџTT
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ў
serving_default
C
input_18
serving_default_input_1:0џџџџџџџџџTT<
sampling0
StatefulPartitionedCall:0џџџџџџџџџ =
	z_log_var0
StatefulPartitionedCall:1џџџџџџџџџ :
z_mean0
StatefulPartitionedCall:2џџџџџџџџџ tensorflow/serving/predict:ы
ѕR
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api

signatures
h__call__
i_default_save_signature
*j&call_and_return_all_conditional_losses"ЈO
_tf_keras_networkO{"class_name": "Functional", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 84, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 84, 84, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Є


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"џ
_tf_keras_layerх{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 84, 4]}}
Њ


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"	
_tf_keras_layerы{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 42, 32]}}
Њ


kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
o__call__
*p&call_and_return_all_conditional_losses"	
_tf_keras_layerы{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 64]}}
т
!regularization_losses
"	variables
#trainable_variables
$	keras_api
q__call__
*r&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Њ

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layerы{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28224}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28224]}}
Ј

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layerщ{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Ў

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layerя{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Е
7regularization_losses
8	variables
9trainable_variables
:	keras_api
y__call__
*z&call_and_return_all_conditional_losses"І
_tf_keras_layer{"class_name": "Sampling", "name": "sampling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling", "trainable": true, "dtype": "float32"}}
K
{0
|1
}2
~3
4
5"
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
Ъ
;layer_regularization_losses
<non_trainable_variables

regularization_losses
	variables

=layers
>layer_metrics
?metrics
trainable_variables
h__call__
i_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
'
{0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
@layer_regularization_losses
Anon_trainable_variables
regularization_losses
	variables

Blayers
Clayer_metrics
Dmetrics
trainable_variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
'
|0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Elayer_regularization_losses
Fnon_trainable_variables
regularization_losses
	variables

Glayers
Hlayer_metrics
Imetrics
trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
'
}0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Jlayer_regularization_losses
Knon_trainable_variables
regularization_losses
	variables

Llayers
Mlayer_metrics
Nmetrics
trainable_variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Olayer_regularization_losses
Pnon_trainable_variables
!regularization_losses
"	variables

Qlayers
Rlayer_metrics
Smetrics
#trainable_variables
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 :
Рм 2dense/kernel
: 2
dense/bias
'
~0"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­
Tlayer_regularization_losses
Unon_trainable_variables
'regularization_losses
(	variables

Vlayers
Wlayer_metrics
Xmetrics
)trainable_variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
:  2z_mean/kernel
: 2z_mean/bias
'
0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
­
Ylayer_regularization_losses
Znon_trainable_variables
-regularization_losses
.	variables

[layers
\layer_metrics
]metrics
/trainable_variables
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
":   2z_log_var/kernel
: 2z_log_var/bias
(
0"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
­
^layer_regularization_losses
_non_trainable_variables
3regularization_losses
4	variables

`layers
alayer_metrics
bmetrics
5trainable_variables
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
clayer_regularization_losses
dnon_trainable_variables
7regularization_losses
8	variables

elayers
flayer_metrics
gmetrics
9trainable_variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
}0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ђ2я
)__inference_encoder_layer_call_fn_1112864
)__inference_encoder_layer_call_fn_1112443
)__inference_encoder_layer_call_fn_1112897
)__inference_encoder_layer_call_fn_1112550Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ш2х
"__inference__wrapped_model_1111973О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџTT
о2л
D__inference_encoder_layer_call_and_return_conditional_losses_1112261
D__inference_encoder_layer_call_and_return_conditional_losses_1112726
D__inference_encoder_layer_call_and_return_conditional_losses_1112831
D__inference_encoder_layer_call_and_return_conditional_losses_1112335Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_conv2d_layer_call_fn_1112929Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_conv2d_layer_call_and_return_conditional_losses_1112920Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_1_layer_call_fn_1112961Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1112952Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_conv2d_2_layer_call_fn_1112993Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1112984Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_layer_call_fn_1113004Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_layer_call_and_return_conditional_losses_1112999Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_layer_call_fn_1113036Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_1113027Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_z_mean_layer_call_fn_1113067Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_z_mean_layer_call_and_return_conditional_losses_1113058Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_z_log_var_layer_call_fn_1113098Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_z_log_var_layer_call_and_return_conditional_losses_1113089Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_sampling_layer_call_fn_1113130Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_sampling_layer_call_and_return_conditional_losses_1113124Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Д2Б
__inference_loss_fn_0_1113141
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_1_1113152
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_2_1113163
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_3_1113174
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_4_1113185
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference_loss_fn_5_1113196
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ЬBЩ
%__inference_signature_wrapper_1112621input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"__inference__wrapped_model_1111973н%&+,128Ђ5
.Ђ+
)&
input_1џџџџџџџџџTT
Њ "Њ
.
sampling"
samplingџџџџџџџџџ 
0
	z_log_var# 
	z_log_varџџџџџџџџџ 
*
z_mean 
z_meanџџџџџџџџџ Е
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1112952l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ** 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_1_layer_call_fn_1112961_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ** 
Њ " џџџџџџџџџ@Е
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1112984l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv2d_2_layer_call_fn_1112993_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Г
C__inference_conv2d_layer_call_and_return_conditional_losses_1112920l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџTT
Њ "-Ђ*
# 
0џџџџџџџџџ** 
 
(__inference_conv2d_layer_call_fn_1112929_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџTT
Њ " џџџџџџџџџ** Є
B__inference_dense_layer_call_and_return_conditional_losses_1113027^%&1Ђ.
'Ђ$
"
inputsџџџџџџџџџРм
Њ "%Ђ"

0џџџџџџџџџ 
 |
'__inference_dense_layer_call_fn_1113036Q%&1Ђ.
'Ђ$
"
inputsџџџџџџџџџРм
Њ "џџџџџџџџџ 
D__inference_encoder_layer_call_and_return_conditional_losses_1112261М%&+,12@Ђ=
6Ђ3
)&
input_1џџџџџџџџџTT
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 
D__inference_encoder_layer_call_and_return_conditional_losses_1112335М%&+,12@Ђ=
6Ђ3
)&
input_1џџџџџџџџџTT
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 
D__inference_encoder_layer_call_and_return_conditional_losses_1112726Л%&+,12?Ђ<
5Ђ2
(%
inputsџџџџџџџџџTT
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 
D__inference_encoder_layer_call_and_return_conditional_losses_1112831Л%&+,12?Ђ<
5Ђ2
(%
inputsџџџџџџџџџTT
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ 

0/1џџџџџџџџџ 

0/2џџџџџџџџџ 
 к
)__inference_encoder_layer_call_fn_1112443Ќ%&+,12@Ђ=
6Ђ3
)&
input_1џџџџџџџџџTT
p

 
Њ "ZW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ к
)__inference_encoder_layer_call_fn_1112550Ќ%&+,12@Ђ=
6Ђ3
)&
input_1џџџџџџџџџTT
p 

 
Њ "ZW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ й
)__inference_encoder_layer_call_fn_1112864Ћ%&+,12?Ђ<
5Ђ2
(%
inputsџџџџџџџџџTT
p

 
Њ "ZW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ й
)__inference_encoder_layer_call_fn_1112897Ћ%&+,12?Ђ<
5Ђ2
(%
inputsџџџџџџџџџTT
p 

 
Њ "ZW

0џџџџџџџџџ 

1џџџџџџџџџ 

2џџџџџџџџџ Њ
D__inference_flatten_layer_call_and_return_conditional_losses_1112999b7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "'Ђ$

0џџџџџџџџџРм
 
)__inference_flatten_layer_call_fn_1113004U7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџРм<
__inference_loss_fn_0_1113141Ђ

Ђ 
Њ " <
__inference_loss_fn_1_1113152Ђ

Ђ 
Њ " <
__inference_loss_fn_2_1113163Ђ

Ђ 
Њ " <
__inference_loss_fn_3_1113174%Ђ

Ђ 
Њ " <
__inference_loss_fn_4_1113185+Ђ

Ђ 
Њ " <
__inference_loss_fn_5_11131961Ђ

Ђ 
Њ " Э
E__inference_sampling_layer_call_and_return_conditional_losses_1113124ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 Є
*__inference_sampling_layer_call_fn_1113130vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ 
Њ "џџџџџџџџџ 
%__inference_signature_wrapper_1112621ш%&+,12CЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџTT"Њ
.
sampling"
samplingџџџџџџџџџ 
0
	z_log_var# 
	z_log_varџџџџџџџџџ 
*
z_mean 
z_meanџџџџџџџџџ І
F__inference_z_log_var_layer_call_and_return_conditional_losses_1113089\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 ~
+__inference_z_log_var_layer_call_fn_1113098O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ Ѓ
C__inference_z_mean_layer_call_and_return_conditional_losses_1113058\+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 {
(__inference_z_mean_layer_call_fn_1113067O+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ 