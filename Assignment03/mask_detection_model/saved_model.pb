??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
?
Conv2D-200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameConv2D-200/kernel
?
%Conv2D-200/kernel/Read/ReadVariableOpReadVariableOpConv2D-200/kernel*'
_output_shapes
:?*
dtype0
w
Conv2D-200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameConv2D-200/bias
p
#Conv2D-200/bias/Read/ReadVariableOpReadVariableOpConv2D-200/bias*
_output_shapes	
:?*
dtype0
?
Conv2D-100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?d*"
shared_nameConv2D-100/kernel
?
%Conv2D-100/kernel/Read/ReadVariableOpReadVariableOpConv2D-100/kernel*'
_output_shapes
:?d*
dtype0
v
Conv2D-100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_nameConv2D-100/bias
o
#Conv2D-100/bias/Read/ReadVariableOpReadVariableOpConv2D-100/bias*
_output_shapes
:d*
dtype0
y
Dense64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N@*
shared_nameDense64/kernel
r
"Dense64/kernel/Read/ReadVariableOpReadVariableOpDense64/kernel*
_output_shapes
:	?N@*
dtype0
p
Dense64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameDense64/bias
i
 Dense64/bias/Read/ReadVariableOpReadVariableOpDense64/bias*
_output_shapes
:@*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:@*
dtype0
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Conv2D-200/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/Conv2D-200/kernel/m
?
,Adam/Conv2D-200/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-200/kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/Conv2D-200/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/Conv2D-200/bias/m
~
*Adam/Conv2D-200/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-200/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Conv2D-100/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?d*)
shared_nameAdam/Conv2D-100/kernel/m
?
,Adam/Conv2D-100/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-100/kernel/m*'
_output_shapes
:?d*
dtype0
?
Adam/Conv2D-100/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_nameAdam/Conv2D-100/bias/m
}
*Adam/Conv2D-100/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-100/bias/m*
_output_shapes
:d*
dtype0
?
Adam/Dense64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N@*&
shared_nameAdam/Dense64/kernel/m
?
)Adam/Dense64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense64/kernel/m*
_output_shapes
:	?N@*
dtype0
~
Adam/Dense64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense64/bias/m
w
'Adam/Dense64/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense64/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/Output/kernel/m
}
(Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/m*
_output_shapes

:@*
dtype0
|
Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/m
u
&Adam/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv2D-200/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/Conv2D-200/kernel/v
?
,Adam/Conv2D-200/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-200/kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/Conv2D-200/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameAdam/Conv2D-200/bias/v
~
*Adam/Conv2D-200/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-200/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Conv2D-100/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?d*)
shared_nameAdam/Conv2D-100/kernel/v
?
,Adam/Conv2D-100/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-100/kernel/v*'
_output_shapes
:?d*
dtype0
?
Adam/Conv2D-100/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_nameAdam/Conv2D-100/bias/v
}
*Adam/Conv2D-100/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-100/bias/v*
_output_shapes
:d*
dtype0
?
Adam/Dense64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N@*&
shared_nameAdam/Dense64/kernel/v
?
)Adam/Dense64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense64/kernel/v*
_output_shapes
:	?N@*
dtype0
~
Adam/Dense64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/Dense64/bias/v
w
'Adam/Dense64/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense64/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/Output/kernel/v
}
(Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/v*
_output_shapes

:@*
dtype0
|
Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/v
u
&Adam/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
8
0
1
2
3
&4
'5
,6
-7
 
8
0
1
2
3
&4
'5
,6
-7
?
7non_trainable_variables
8layer_metrics
		variables
9layer_regularization_losses
:metrics

;layers

regularization_losses
trainable_variables
 
][
VARIABLE_VALUEConv2D-200/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEConv2D-200/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
<non_trainable_variables
	variables
=layer_regularization_losses
>metrics

?layers
regularization_losses
@layer_metrics
trainable_variables
 
 
 
?
Anon_trainable_variables
	variables
Blayer_regularization_losses
Cmetrics

Dlayers
regularization_losses
Elayer_metrics
trainable_variables
][
VARIABLE_VALUEConv2D-100/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEConv2D-100/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Fnon_trainable_variables
	variables
Glayer_regularization_losses
Hmetrics

Ilayers
regularization_losses
Jlayer_metrics
trainable_variables
 
 
 
?
Knon_trainable_variables
	variables
Llayer_regularization_losses
Mmetrics

Nlayers
regularization_losses
Olayer_metrics
 trainable_variables
 
 
 
?
Pnon_trainable_variables
"	variables
Qlayer_regularization_losses
Rmetrics

Slayers
#regularization_losses
Tlayer_metrics
$trainable_variables
ZX
VARIABLE_VALUEDense64/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense64/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
?
Unon_trainable_variables
(	variables
Vlayer_regularization_losses
Wmetrics

Xlayers
)regularization_losses
Ylayer_metrics
*trainable_variables
YW
VARIABLE_VALUEOutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEOutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
Znon_trainable_variables
.	variables
[layer_regularization_losses
\metrics

]layers
/regularization_losses
^layer_metrics
0trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

_0
`1
1
0
1
2
3
4
5
6
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
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
?~
VARIABLE_VALUEAdam/Conv2D-200/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv2D-200/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Conv2D-100/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv2D-100/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense64/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense64/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Conv2D-200/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv2D-200/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Conv2D-100/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Conv2D-100/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense64/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense64/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_InputPlaceholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputConv2D-200/kernelConv2D-200/biasConv2D-100/kernelConv2D-100/biasDense64/kernelDense64/biasOutput/kernelOutput/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_5152
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%Conv2D-200/kernel/Read/ReadVariableOp#Conv2D-200/bias/Read/ReadVariableOp%Conv2D-100/kernel/Read/ReadVariableOp#Conv2D-100/bias/Read/ReadVariableOp"Dense64/kernel/Read/ReadVariableOp Dense64/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/Conv2D-200/kernel/m/Read/ReadVariableOp*Adam/Conv2D-200/bias/m/Read/ReadVariableOp,Adam/Conv2D-100/kernel/m/Read/ReadVariableOp*Adam/Conv2D-100/bias/m/Read/ReadVariableOp)Adam/Dense64/kernel/m/Read/ReadVariableOp'Adam/Dense64/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp,Adam/Conv2D-200/kernel/v/Read/ReadVariableOp*Adam/Conv2D-200/bias/v/Read/ReadVariableOp,Adam/Conv2D-100/kernel/v/Read/ReadVariableOp*Adam/Conv2D-100/bias/v/Read/ReadVariableOp)Adam/Dense64/kernel/v/Read/ReadVariableOp'Adam/Dense64/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_5479
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv2D-200/kernelConv2D-200/biasConv2D-100/kernelConv2D-100/biasDense64/kernelDense64/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv2D-200/kernel/mAdam/Conv2D-200/bias/mAdam/Conv2D-100/kernel/mAdam/Conv2D-100/bias/mAdam/Dense64/kernel/mAdam/Dense64/bias/mAdam/Output/kernel/mAdam/Output/bias/mAdam/Conv2D-200/kernel/vAdam/Conv2D-200/bias/vAdam/Conv2D-100/kernel/vAdam/Conv2D-100/bias/vAdam/Dense64/kernel/vAdam/Dense64/bias/vAdam/Output/kernel/vAdam/Output/bias/v*-
Tin&
$2"*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_5588??
?
?
A__inference_Dense64_layer_call_and_return_conditional_losses_4953

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?N@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????N:::P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?
?
A__inference_Dense64_layer_call_and_return_conditional_losses_5328

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?N@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????N:::P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5054

inputs
conv2d_200_5030
conv2d_200_5032
conv2d_100_5036
conv2d_100_5038
dense64_5043
dense64_5045
output_5048
output_5050
identity??"Conv2D-100/StatefulPartitionedCall?"Conv2D-200/StatefulPartitionedCall?Dense64/StatefulPartitionedCall?Output/StatefulPartitionedCall?
"Conv2D-200/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_200_5030conv2d_200_5032*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bb?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_48832$
"Conv2D-200/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall+Conv2D-200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_48502
max_pooling2d/PartitionedCall?
"Conv2D-100/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_100_5036conv2d_100_5038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_49112$
"Conv2D-100/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall+Conv2D-100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48622!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????N* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_49342
flatten/PartitionedCall?
Dense64/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense64_5043dense64_5045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense64_layer_call_and_return_conditional_losses_49532!
Dense64/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense64/StatefulPartitionedCall:output:0output_5048output_5050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_49802 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0#^Conv2D-100/StatefulPartitionedCall#^Conv2D-200/StatefulPartitionedCall ^Dense64/StatefulPartitionedCall^Output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::2H
"Conv2D-100/StatefulPartitionedCall"Conv2D-100/StatefulPartitionedCall2H
"Conv2D-200/StatefulPartitionedCall"Conv2D-200/StatefulPartitionedCall2B
Dense64/StatefulPartitionedCallDense64/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_4856

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_48502
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5024	
input
conv2d_200_5000
conv2d_200_5002
conv2d_100_5006
conv2d_100_5008
dense64_5013
dense64_5015
output_5018
output_5020
identity??"Conv2D-100/StatefulPartitionedCall?"Conv2D-200/StatefulPartitionedCall?Dense64/StatefulPartitionedCall?Output/StatefulPartitionedCall?
"Conv2D-200/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_200_5000conv2d_200_5002*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bb?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_48832$
"Conv2D-200/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall+Conv2D-200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_48502
max_pooling2d/PartitionedCall?
"Conv2D-100/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_100_5006conv2d_100_5008*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_49112$
"Conv2D-100/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall+Conv2D-100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48622!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????N* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_49342
flatten/PartitionedCall?
Dense64/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense64_5013dense64_5015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense64_layer_call_and_return_conditional_losses_49532!
Dense64/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense64/StatefulPartitionedCall:output:0output_5018output_5020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_49802 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0#^Conv2D-100/StatefulPartitionedCall#^Conv2D-200/StatefulPartitionedCall ^Dense64/StatefulPartitionedCall^Output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::2H
"Conv2D-100/StatefulPartitionedCall"Conv2D-100/StatefulPartitionedCall2H
"Conv2D-200/StatefulPartitionedCall"Conv2D-200/StatefulPartitionedCall2B
Dense64/StatefulPartitionedCallDense64/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?
B
&__inference_flatten_layer_call_fn_5317

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????N* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_49342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

d:W S
/
_output_shapes
:?????????

d
 
_user_specified_nameinputs
?
?
@__inference_Output_layer_call_and_return_conditional_losses_4980

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
~
)__inference_Conv2D-200_layer_call_fn_5286

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bb?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_48832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????bb?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????dd::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_4997	
input
conv2d_200_4894
conv2d_200_4896
conv2d_100_4922
conv2d_100_4924
dense64_4964
dense64_4966
output_4991
output_4993
identity??"Conv2D-100/StatefulPartitionedCall?"Conv2D-200/StatefulPartitionedCall?Dense64/StatefulPartitionedCall?Output/StatefulPartitionedCall?
"Conv2D-200/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_200_4894conv2d_200_4896*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bb?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_48832$
"Conv2D-200/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall+Conv2D-200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_48502
max_pooling2d/PartitionedCall?
"Conv2D-100/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_100_4922conv2d_100_4924*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_49112$
"Conv2D-100/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall+Conv2D-100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48622!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????N* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_49342
flatten/PartitionedCall?
Dense64/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense64_4964dense64_4966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense64_layer_call_and_return_conditional_losses_49532!
Dense64/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense64/StatefulPartitionedCall:output:0output_4991output_4993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_49802 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0#^Conv2D-100/StatefulPartitionedCall#^Conv2D-200/StatefulPartitionedCall ^Dense64/StatefulPartitionedCall^Output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::2H
"Conv2D-100/StatefulPartitionedCall"Conv2D-100/StatefulPartitionedCall2H
"Conv2D-200/StatefulPartitionedCall"Conv2D-200/StatefulPartitionedCall2B
Dense64/StatefulPartitionedCallDense64/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?
z
%__inference_Output_layer_call_fn_5357

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_49802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_5245

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_50542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4862

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_5312

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????'  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????N2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

d:W S
/
_output_shapes
:?????????

d
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_5152	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_48442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4850

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_5266

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_51022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?	
?
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_4911

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????d2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ?:::X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?	
?
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_5277

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????bb?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????bb?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????dd:::W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?,
?
__inference__wrapped_model_4844	
input8
4sequential_conv2d_200_conv2d_readvariableop_resource9
5sequential_conv2d_200_biasadd_readvariableop_resource8
4sequential_conv2d_100_conv2d_readvariableop_resource9
5sequential_conv2d_100_biasadd_readvariableop_resource5
1sequential_dense64_matmul_readvariableop_resource6
2sequential_dense64_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity??
+sequential/Conv2D-200/Conv2D/ReadVariableOpReadVariableOp4sequential_conv2d_200_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02-
+sequential/Conv2D-200/Conv2D/ReadVariableOp?
sequential/Conv2D-200/Conv2DConv2Dinput3sequential/Conv2D-200/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?*
paddingVALID*
strides
2
sequential/Conv2D-200/Conv2D?
,sequential/Conv2D-200/BiasAdd/ReadVariableOpReadVariableOp5sequential_conv2d_200_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential/Conv2D-200/BiasAdd/ReadVariableOp?
sequential/Conv2D-200/BiasAddBiasAdd%sequential/Conv2D-200/Conv2D:output:04sequential/Conv2D-200/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?2
sequential/Conv2D-200/BiasAdd?
sequential/Conv2D-200/ReluRelu&sequential/Conv2D-200/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bb?2
sequential/Conv2D-200/Relu?
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/Conv2D-200/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
+sequential/Conv2D-100/Conv2D/ReadVariableOpReadVariableOp4sequential_conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype02-
+sequential/Conv2D-100/Conv2D/ReadVariableOp?
sequential/Conv2D-100/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:03sequential/Conv2D-100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
sequential/Conv2D-100/Conv2D?
,sequential/Conv2D-100/BiasAdd/ReadVariableOpReadVariableOp5sequential_conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,sequential/Conv2D-100/BiasAdd/ReadVariableOp?
sequential/Conv2D-100/BiasAddBiasAdd%sequential/Conv2D-100/Conv2D:output:04sequential/Conv2D-100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2
sequential/Conv2D-100/BiasAdd?
sequential/Conv2D-100/ReluRelu&sequential/Conv2D-100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????d2
sequential/Conv2D-100/Relu?
"sequential/max_pooling2d_1/MaxPoolMaxPool(sequential/Conv2D-100/Relu:activations:0*/
_output_shapes
:?????????

d*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????'  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????N2
sequential/flatten/Reshape?
(sequential/Dense64/MatMul/ReadVariableOpReadVariableOp1sequential_dense64_matmul_readvariableop_resource*
_output_shapes
:	?N@*
dtype02*
(sequential/Dense64/MatMul/ReadVariableOp?
sequential/Dense64/MatMulMatMul#sequential/flatten/Reshape:output:00sequential/Dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/Dense64/MatMul?
)sequential/Dense64/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/Dense64/BiasAdd/ReadVariableOp?
sequential/Dense64/BiasAddBiasAdd#sequential/Dense64/MatMul:product:01sequential/Dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/Dense64/BiasAdd?
sequential/Dense64/ReluRelu#sequential/Dense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential/Dense64/Relu?
'sequential/Output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02)
'sequential/Output/MatMul/ReadVariableOp?
sequential/Output/MatMulMatMul%sequential/Dense64/Relu:activations:0/sequential/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/Output/MatMul?
(sequential/Output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/Output/BiasAdd/ReadVariableOp?
sequential/Output/BiasAddBiasAdd"sequential/Output/MatMul:product:00sequential/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/Output/BiasAdd?
sequential/Output/SoftmaxSoftmax"sequential/Output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/Output/Softmaxw
IdentityIdentity#sequential/Output/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd:::::::::V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?
~
)__inference_Conv2D-100_layer_call_fn_5306

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_49112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_5121	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_51022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?
{
&__inference_Dense64_layer_call_fn_5337

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense64_layer_call_and_return_conditional_losses_49532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????N::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5102

inputs
conv2d_200_5078
conv2d_200_5080
conv2d_100_5084
conv2d_100_5086
dense64_5091
dense64_5093
output_5096
output_5098
identity??"Conv2D-100/StatefulPartitionedCall?"Conv2D-200/StatefulPartitionedCall?Dense64/StatefulPartitionedCall?Output/StatefulPartitionedCall?
"Conv2D-200/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_200_5078conv2d_200_5080*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????bb?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_48832$
"Conv2D-200/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall+Conv2D-200/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_48502
max_pooling2d/PartitionedCall?
"Conv2D-100/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_100_5084conv2d_100_5086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_49112$
"Conv2D-100/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall+Conv2D-100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48622!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????N* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_49342
flatten/PartitionedCall?
Dense64/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense64_5091dense64_5093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense64_layer_call_and_return_conditional_losses_49532!
Dense64/StatefulPartitionedCall?
Output/StatefulPartitionedCallStatefulPartitionedCall(Dense64/StatefulPartitionedCall:output:0output_5096output_5098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Output_layer_call_and_return_conditional_losses_49802 
Output/StatefulPartitionedCall?
IdentityIdentity'Output/StatefulPartitionedCall:output:0#^Conv2D-100/StatefulPartitionedCall#^Conv2D-200/StatefulPartitionedCall ^Dense64/StatefulPartitionedCall^Output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::2H
"Conv2D-100/StatefulPartitionedCall"Conv2D-100/StatefulPartitionedCall2H
"Conv2D-200/StatefulPartitionedCall"Conv2D-200/StatefulPartitionedCall2B
Dense64/StatefulPartitionedCallDense64/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_5588
file_prefix&
"assignvariableop_conv2d_200_kernel&
"assignvariableop_1_conv2d_200_bias(
$assignvariableop_2_conv2d_100_kernel&
"assignvariableop_3_conv2d_100_bias%
!assignvariableop_4_dense64_kernel#
assignvariableop_5_dense64_bias$
 assignvariableop_6_output_kernel"
assignvariableop_7_output_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_10
,assignvariableop_17_adam_conv2d_200_kernel_m.
*assignvariableop_18_adam_conv2d_200_bias_m0
,assignvariableop_19_adam_conv2d_100_kernel_m.
*assignvariableop_20_adam_conv2d_100_bias_m-
)assignvariableop_21_adam_dense64_kernel_m+
'assignvariableop_22_adam_dense64_bias_m,
(assignvariableop_23_adam_output_kernel_m*
&assignvariableop_24_adam_output_bias_m0
,assignvariableop_25_adam_conv2d_200_kernel_v.
*assignvariableop_26_adam_conv2d_200_bias_v0
,assignvariableop_27_adam_conv2d_100_kernel_v.
*assignvariableop_28_adam_conv2d_100_bias_v-
)assignvariableop_29_adam_dense64_kernel_v+
'assignvariableop_30_adam_dense64_bias_v,
(assignvariableop_31_adam_output_kernel_v*
&assignvariableop_32_adam_output_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_200_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_200_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_100_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_100_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense64_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense64_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_conv2d_200_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_conv2d_200_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_100_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_100_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense64_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense64_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_conv2d_200_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv2d_200_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_conv2d_100_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_100_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense64_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense64_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_output_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_output_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_5188

inputs-
)conv2d_200_conv2d_readvariableop_resource.
*conv2d_200_biasadd_readvariableop_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource*
&dense64_matmul_readvariableop_resource+
'dense64_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
 Conv2D-200/Conv2D/ReadVariableOpReadVariableOp)conv2d_200_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02"
 Conv2D-200/Conv2D/ReadVariableOp?
Conv2D-200/Conv2DConv2Dinputs(Conv2D-200/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?*
paddingVALID*
strides
2
Conv2D-200/Conv2D?
!Conv2D-200/BiasAdd/ReadVariableOpReadVariableOp*conv2d_200_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!Conv2D-200/BiasAdd/ReadVariableOp?
Conv2D-200/BiasAddBiasAddConv2D-200/Conv2D:output:0)Conv2D-200/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?2
Conv2D-200/BiasAdd?
Conv2D-200/ReluReluConv2D-200/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bb?2
Conv2D-200/Relu?
max_pooling2d/MaxPoolMaxPoolConv2D-200/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
 Conv2D-100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype02"
 Conv2D-100/Conv2D/ReadVariableOp?
Conv2D-100/Conv2DConv2Dmax_pooling2d/MaxPool:output:0(Conv2D-100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
Conv2D-100/Conv2D?
!Conv2D-100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02#
!Conv2D-100/BiasAdd/ReadVariableOp?
Conv2D-100/BiasAddBiasAddConv2D-100/Conv2D:output:0)Conv2D-100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2
Conv2D-100/BiasAdd?
Conv2D-100/ReluReluConv2D-100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????d2
Conv2D-100/Relu?
max_pooling2d_1/MaxPoolMaxPoolConv2D-100/Relu:activations:0*/
_output_shapes
:?????????

d*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????'  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????N2
flatten/Reshape?
Dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes
:	?N@*
dtype02
Dense64/MatMul/ReadVariableOp?
Dense64/MatMulMatMulflatten/Reshape:output:0%Dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Dense64/MatMul?
Dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
Dense64/BiasAdd/ReadVariableOp?
Dense64/BiasAddBiasAddDense64/MatMul:product:0&Dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Dense64/BiasAddp
Dense64/ReluReluDense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Dense64/Relu?
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Output/MatMul/ReadVariableOp?
Output/MatMulMatMulDense64/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Output/MatMul?
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Output/BiasAddv
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Output/Softmaxl
IdentityIdentityOutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd:::::::::W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_4934

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????'  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????N2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????N2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????

d:W S
/
_output_shapes
:?????????

d
 
_user_specified_nameinputs
?	
?
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_4883

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????bb?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????bb?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????dd:::W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
@__inference_Output_layer_call_and_return_conditional_losses_5348

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_5073	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_50542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????dd

_user_specified_nameInput
?	
?
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_5297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????d2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????  ?:::X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_1_layer_call_fn_4868

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_48622
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?H
?
__inference__traced_save_5479
file_prefix0
,savev2_conv2d_200_kernel_read_readvariableop.
*savev2_conv2d_200_bias_read_readvariableop0
,savev2_conv2d_100_kernel_read_readvariableop.
*savev2_conv2d_100_bias_read_readvariableop-
)savev2_dense64_kernel_read_readvariableop+
'savev2_dense64_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_200_kernel_m_read_readvariableop5
1savev2_adam_conv2d_200_bias_m_read_readvariableop7
3savev2_adam_conv2d_100_kernel_m_read_readvariableop5
1savev2_adam_conv2d_100_bias_m_read_readvariableop4
0savev2_adam_dense64_kernel_m_read_readvariableop2
.savev2_adam_dense64_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop7
3savev2_adam_conv2d_200_kernel_v_read_readvariableop5
1savev2_adam_conv2d_200_bias_v_read_readvariableop7
3savev2_adam_conv2d_100_kernel_v_read_readvariableop5
1savev2_adam_conv2d_100_bias_v_read_readvariableop4
0savev2_adam_dense64_kernel_v_read_readvariableop2
.savev2_adam_dense64_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e6b674ab56bb499d8295e3c878205a83/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_200_kernel_read_readvariableop*savev2_conv2d_200_bias_read_readvariableop,savev2_conv2d_100_kernel_read_readvariableop*savev2_conv2d_100_bias_read_readvariableop)savev2_dense64_kernel_read_readvariableop'savev2_dense64_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_200_kernel_m_read_readvariableop1savev2_adam_conv2d_200_bias_m_read_readvariableop3savev2_adam_conv2d_100_kernel_m_read_readvariableop1savev2_adam_conv2d_100_bias_m_read_readvariableop0savev2_adam_dense64_kernel_m_read_readvariableop.savev2_adam_dense64_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop3savev2_adam_conv2d_200_kernel_v_read_readvariableop1savev2_adam_conv2d_200_bias_v_read_readvariableop3savev2_adam_conv2d_100_kernel_v_read_readvariableop1savev2_adam_conv2d_100_bias_v_read_readvariableop0savev2_adam_dense64_kernel_v_read_readvariableop.savev2_adam_dense64_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?d:d:	?N@:@:@:: : : : : : : : : :?:?:?d:d:	?N@:@:@::?:?:?d:d:	?N@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?d: 

_output_shapes
:d:%!

_output_shapes
:	?N@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?d: 

_output_shapes
:d:%!

_output_shapes
:	?N@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::-)
'
_output_shapes
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?d: 

_output_shapes
:d:%!

_output_shapes
:	?N@: 

_output_shapes
:@:$  

_output_shapes

:@: !

_output_shapes
::"

_output_shapes
: 
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_5224

inputs-
)conv2d_200_conv2d_readvariableop_resource.
*conv2d_200_biasadd_readvariableop_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource*
&dense64_matmul_readvariableop_resource+
'dense64_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
 Conv2D-200/Conv2D/ReadVariableOpReadVariableOp)conv2d_200_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02"
 Conv2D-200/Conv2D/ReadVariableOp?
Conv2D-200/Conv2DConv2Dinputs(Conv2D-200/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?*
paddingVALID*
strides
2
Conv2D-200/Conv2D?
!Conv2D-200/BiasAdd/ReadVariableOpReadVariableOp*conv2d_200_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!Conv2D-200/BiasAdd/ReadVariableOp?
Conv2D-200/BiasAddBiasAddConv2D-200/Conv2D:output:0)Conv2D-200/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????bb?2
Conv2D-200/BiasAdd?
Conv2D-200/ReluReluConv2D-200/BiasAdd:output:0*
T0*0
_output_shapes
:?????????bb?2
Conv2D-200/Relu?
max_pooling2d/MaxPoolMaxPoolConv2D-200/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
 Conv2D-100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype02"
 Conv2D-100/Conv2D/ReadVariableOp?
Conv2D-100/Conv2DConv2Dmax_pooling2d/MaxPool:output:0(Conv2D-100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingVALID*
strides
2
Conv2D-100/Conv2D?
!Conv2D-100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02#
!Conv2D-100/BiasAdd/ReadVariableOp?
Conv2D-100/BiasAddBiasAddConv2D-100/Conv2D:output:0)Conv2D-100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d2
Conv2D-100/BiasAdd?
Conv2D-100/ReluReluConv2D-100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????d2
Conv2D-100/Relu?
max_pooling2d_1/MaxPoolMaxPoolConv2D-100/Relu:activations:0*/
_output_shapes
:?????????

d*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????'  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????N2
flatten/Reshape?
Dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes
:	?N@*
dtype02
Dense64/MatMul/ReadVariableOp?
Dense64/MatMulMatMulflatten/Reshape:output:0%Dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Dense64/MatMul?
Dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
Dense64/BiasAdd/ReadVariableOp?
Dense64/BiasAddBiasAddDense64/MatMul:product:0&Dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Dense64/BiasAddp
Dense64/ReluReluDense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Dense64/Relu?
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
Output/MatMul/ReadVariableOp?
Output/MatMulMatMulDense64/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Output/MatMul?
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
Output/BiasAdd/ReadVariableOp?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Output/BiasAddv
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Output/Softmaxl
IdentityIdentityOutput/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????dd:::::::::W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
Input6
serving_default_Input:0?????????dd:
Output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?;
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
z_default_save_signature
*{&call_and_return_all_conditional_losses
|__call__"?8
_tf_keras_sequential?7{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D-200", "trainable": true, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D-100", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D-200", "trainable": true, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv2D-100", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Conv2D-200", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D-200", "trainable": true, "dtype": "float32", "filters": 200, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 1]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Conv2D-100", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D-100", "trainable": true, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 200]}}
?
	variables
regularization_losses
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10000]}}
?

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
?
7non_trainable_variables
8layer_metrics
		variables
9layer_regularization_losses
:metrics

;layers

regularization_losses
trainable_variables
|__call__
z_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
,:*?2Conv2D-200/kernel
:?2Conv2D-200/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
<non_trainable_variables
	variables
=layer_regularization_losses
>metrics

?layers
regularization_losses
@layer_metrics
trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables
	variables
Blayer_regularization_losses
Cmetrics

Dlayers
regularization_losses
Elayer_metrics
trainable_variables
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,:*?d2Conv2D-100/kernel
:d2Conv2D-100/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Fnon_trainable_variables
	variables
Glayer_regularization_losses
Hmetrics

Ilayers
regularization_losses
Jlayer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables
	variables
Llayer_regularization_losses
Mmetrics

Nlayers
regularization_losses
Olayer_metrics
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables
"	variables
Qlayer_regularization_losses
Rmetrics

Slayers
#regularization_losses
Tlayer_metrics
$trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?N@2Dense64/kernel
:@2Dense64/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
Unon_trainable_variables
(	variables
Vlayer_regularization_losses
Wmetrics

Xlayers
)regularization_losses
Ylayer_metrics
*trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@2Output/kernel
:2Output/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
Znon_trainable_variables
.	variables
[layer_regularization_losses
\metrics

]layers
/regularization_losses
^layer_metrics
0trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	atotal
	bcount
c	variables
d	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
1:/?2Adam/Conv2D-200/kernel/m
#:!?2Adam/Conv2D-200/bias/m
1:/?d2Adam/Conv2D-100/kernel/m
": d2Adam/Conv2D-100/bias/m
&:$	?N@2Adam/Dense64/kernel/m
:@2Adam/Dense64/bias/m
$:"@2Adam/Output/kernel/m
:2Adam/Output/bias/m
1:/?2Adam/Conv2D-200/kernel/v
#:!?2Adam/Conv2D-200/bias/v
1:/?d2Adam/Conv2D-100/kernel/v
": d2Adam/Conv2D-100/bias/v
&:$	?N@2Adam/Dense64/kernel/v
:@2Adam/Dense64/bias/v
$:"@2Adam/Output/kernel/v
:2Adam/Output/bias/v
?2?
__inference__wrapped_model_4844?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *,?)
'?$
Input?????????dd
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_5024
D__inference_sequential_layer_call_and_return_conditional_losses_4997
D__inference_sequential_layer_call_and_return_conditional_losses_5224
D__inference_sequential_layer_call_and_return_conditional_losses_5188?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_5245
)__inference_sequential_layer_call_fn_5266
)__inference_sequential_layer_call_fn_5073
)__inference_sequential_layer_call_fn_5121?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_5277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Conv2D-200_layer_call_fn_5286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4850?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_max_pooling2d_layer_call_fn_4856?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_5297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Conv2D-100_layer_call_fn_5306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_1_layer_call_fn_4868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_5312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_5317?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Dense64_layer_call_and_return_conditional_losses_5328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense64_layer_call_fn_5337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Output_layer_call_and_return_conditional_losses_5348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_Output_layer_call_fn_5357?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
/B-
"__inference_signature_wrapper_5152Input?
D__inference_Conv2D-100_layer_call_and_return_conditional_losses_5297m8?5
.?+
)?&
inputs?????????  ?
? "-?*
#? 
0?????????d
? ?
)__inference_Conv2D-100_layer_call_fn_5306`8?5
.?+
)?&
inputs?????????  ?
? " ??????????d?
D__inference_Conv2D-200_layer_call_and_return_conditional_losses_5277m7?4
-?*
(?%
inputs?????????dd
? ".?+
$?!
0?????????bb?
? ?
)__inference_Conv2D-200_layer_call_fn_5286`7?4
-?*
(?%
inputs?????????dd
? "!??????????bb??
A__inference_Dense64_layer_call_and_return_conditional_losses_5328]&'0?-
&?#
!?
inputs??????????N
? "%?"
?
0?????????@
? z
&__inference_Dense64_layer_call_fn_5337P&'0?-
&?#
!?
inputs??????????N
? "??????????@?
@__inference_Output_layer_call_and_return_conditional_losses_5348\,-/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? x
%__inference_Output_layer_call_fn_5357O,-/?,
%?"
 ?
inputs?????????@
? "???????????
__inference__wrapped_model_4844s&',-6?3
,?)
'?$
Input?????????dd
? "/?,
*
Output ?
Output??????????
A__inference_flatten_layer_call_and_return_conditional_losses_5312a7?4
-?*
(?%
inputs?????????

d
? "&?#
?
0??????????N
? ~
&__inference_flatten_layer_call_fn_5317T7?4
-?*
(?%
inputs?????????

d
? "???????????N?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4862?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_1_layer_call_fn_4868?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4850?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling2d_layer_call_fn_4856?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_sequential_layer_call_and_return_conditional_losses_4997q&',->?;
4?1
'?$
Input?????????dd
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_5024q&',->?;
4?1
'?$
Input?????????dd
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_5188r&',-??<
5?2
(?%
inputs?????????dd
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_5224r&',-??<
5?2
(?%
inputs?????????dd
p 

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_5073d&',->?;
4?1
'?$
Input?????????dd
p

 
? "???????????
)__inference_sequential_layer_call_fn_5121d&',->?;
4?1
'?$
Input?????????dd
p 

 
? "???????????
)__inference_sequential_layer_call_fn_5245e&',-??<
5?2
(?%
inputs?????????dd
p

 
? "???????????
)__inference_sequential_layer_call_fn_5266e&',-??<
5?2
(?%
inputs?????????dd
p 

 
? "???????????
"__inference_signature_wrapper_5152|&',-??<
? 
5?2
0
Input'?$
Input?????????dd"/?,
*
Output ?
Output?????????