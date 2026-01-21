"""Demonstrates lowerings for different styles of compile and torchfunctional.

Log output below, indicating the compiled train step DOES have a graph break
in the dynamo graph before lowering to the AOTAutograd graph.

However, the torch functional style does NOT have a graph break.

```
(tt) yho_google_com@yho-l4:~/Documents/GitHub/sandbox$ TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_LOGS="graph_code, graph_breaks, compiled_autograd, aot_graphs" python scripts/example_compile_lowerings.py





Style 1: eager.





Style 2: torch.compile on the module.
/opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/pgo.py:537: UserWarning: dynamo_pgo force disabled by torch.compiler.config.force_disable_caches
  warn_once(
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code] TRACED GRAPH
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]  ===== __compiled_fn_1_ed0bf192_320e_48ac_9f71_cfda62d35d5b =====
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]     def forward(self, L_self_parameters_scale_: "f32[1][1]cpu", L_x_: "f32[8, 256][256, 1]cpu"):
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         l_self_parameters_scale_ = L_self_parameters_scale_
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         l_x_ = L_x_
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = l_self_parameters_scale_ * l_x_;  l_self_parameters_scale_ = l_x_ = None
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         sin: "f32[8, 256][256, 1]cpu" = mul.sin();  mul = None
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         return (sin,)
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
V0121 00:08:38.912000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [0/0] [__aot_graphs] aot_config id: 0, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]  ===== Forward graph 0 =====
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu"):
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         sin: "f32[8, 256][256, 1]cpu" = torch.ops.aten.sin.default(mul);  mul = None
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         return (sin, primals_1, primals_2)
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]
I0121 00:08:39.321000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]  ===== Backward graph 0 =====
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]  <eval_with_key>.2 class GraphModule(torch.nn.Module):
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", tangents_1: "f32[8, 256][256, 1]cpu"):
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(mul);  mul = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, primals_1);  primals_1 = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul_3: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         sum_1: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True);  mul_3 = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         return (view, mul_2)
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]
I0121 00:08:39.322000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]





Style 3: torch.compile on a compiled train_step().
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] Graph break in user code at /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:58
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] Graph Break Reason: Unsupported Tensor.backward() call
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Explanation: Dynamo currently does not support tracing `Tensor.backward()`.
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Developer debug context: call_method TensorVariable() backward () {}
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0123.html
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] User code traceback:
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 104, in <module>
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     sys.exit(main())
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 98, in main
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     compiled_train_step(Model(), data)
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 64, in compiled_train_step
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 58, in train_step
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     loss.backward()
V0121 00:08:50.929000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code] TRACED GRAPH
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]  ===== __compiled_fn_4_7dad6abf_5b1f_4f69_a03d_b9d931cab930 =====
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]     def forward(self, L_model_parameters_scale_: "f32[1][1]cpu", L_data_: "f32[8, 256][256, 1]cpu"):
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         l_model_parameters_scale_ = L_model_parameters_scale_
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         l_data_ = L_data_
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = l_model_parameters_scale_ * l_data_;  l_model_parameters_scale_ = l_data_ = None
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         out: "f32[8, 256][256, 1]cpu" = mul.sin();  mul = None
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         loss: "f32[][]cpu" = out.sum()
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         return (loss, out)
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0121 00:08:50.941000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
I0121 00:08:51.000000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [1/0_1] [__aot_graphs] aot_config id: 1, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=()), FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0)), TangentAOTInput(output=PlainAOTOutput(idx=1))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format)), PlainTensorMeta(unwrapped_idx=1, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=()), FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0)), TangentAOTInput(output=PlainAOTOutput(idx=1))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format)), PlainTensorMeta(unwrapped_idx=1, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs] TRACED GRAPH
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]  ===== Forward graph 1 =====
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu"):
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         sin: "f32[8, 256][256, 1]cpu" = torch.ops.aten.sin.default(mul);  mul = None
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         sum_1: "f32[][]cpu" = torch.ops.aten.sum.default(sin)
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         return (sum_1, sin, primals_1, primals_2)
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0121 00:08:51.001000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs] TRACED GRAPH
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]  ===== Backward graph 1 =====
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]  <eval_with_key>.8 class GraphModule(torch.nn.Module):
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", tangents_1: "f32[][]cpu", tangents_2: "f32[8, 256][256, 1]cpu"):
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(tangents_1, [8, 256]);  tangents_1 = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(tangents_2, expand);  tangents_2 = expand = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(mul);  mul = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(add, cos);  add = cos = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, primals_1);  primals_1 = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul_3: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True);  mul_3 = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         return (view, mul_2)
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0121 00:08:51.002000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd] TRACED GRAPH
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]  ===== Compiled autograd graph =====
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]  <eval_with_key>.10 class CompiledAutograd0(torch.nn.Module):
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]     def forward(self, inputs, sizes, scalars, hooks, packed_data):
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         # No stacktrace found for following nodes
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem = inputs[0]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_1 = inputs[1]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_2 = inputs[2]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_3 = inputs[3]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_4 = inputs[4];  inputs = getitem_4 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: torch::autograd::GraphRoot (NodeCall 0)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_5 = validate_outputs[0];  validate_outputs = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: CompiledFunctionBackward1 (NodeCall 1)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         zeros = torch.ops.aten.zeros.default([8, 256], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_6 = hooks[0];  hooks = getitem_6 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((getitem_1, getitem_2), [], getitem_5, zeros);  getitem_2 = getitem_5 = zeros = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_primals_1 = call_aot_bwd_prologue[0]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_primals_2 = call_aot_bwd_prologue[1]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_tangents_1 = call_aot_bwd_prologue[2]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_tangents_2 = call_aot_bwd_prologue[3];  call_aot_bwd_prologue = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_expand = torch.ops.aten.expand.default(aot1_tangents_1, [8, 256]);  aot1_tangents_1 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_add = torch.ops.aten.add.Tensor(aot1_tangents_2, aot1_expand);  aot1_tangents_2 = aot1_expand = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_mul = torch.ops.aten.mul.Tensor(aot1_primals_1, aot1_primals_2)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_cos = torch.ops.aten.cos.default(aot1_mul);  aot1_mul = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_mul_1 = torch.ops.aten.mul.Tensor(aot1_add, aot1_cos);  aot1_add = aot1_cos = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_mul_2 = torch.ops.aten.mul.Tensor(aot1_mul_1, aot1_primals_1);  aot1_primals_1 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_mul_3 = torch.ops.aten.mul.Tensor(aot1_mul_1, aot1_primals_2);  aot1_mul_1 = aot1_primals_2 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_sum_2 = torch.ops.aten.sum.dim_IntList(aot1_mul_3, [0, 1], True);  aot1_mul_3 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         aot1_view = torch.ops.aten.reshape.default(aot1_sum_2, [1]);  aot1_sum_2 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: torch::autograd::AccumulateGrad (NodeCall 4)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         call_accumulate_grad_1 = torch__dynamo_external_utils_call_accumulate_grad(getitem_1, aot1_view, False);  getitem_1 = aot1_view = call_accumulate_grad_1 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: ViewBackward0 (NodeCall 2)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         view_backward0 = torch__dynamo_compiled_autograd_ops_ViewBackward0([aot1_mul_2], [True], [2048]);  aot1_mul_2 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_11 = view_backward0[0];  view_backward0 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11], [((None, None, device(type='cpu'), 6, 0, None), [2048], False)]);  getitem_11 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         getitem_12 = validate_outputs_1[0];  validate_outputs_1 = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: torch::autograd::AccumulateGrad (NodeCall 3)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         call_accumulate_grad = torch__dynamo_external_utils_call_accumulate_grad(getitem_3, getitem_12, False);  getitem_3 = getitem_12 = call_accumulate_grad = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: torch::autograd::AccumulateGrad (NodeCall 4)
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         _exec_final_callbacks_stub = torch__dynamo_external_utils__exec_final_callbacks_stub();  _exec_final_callbacks_stub = None
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]         return []
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
I0121 00:08:52.348000 2403842 site-packages/torch/_dynamo/compiled_autograd.py:1096] [!0] [__compiled_autograd]
W0121 00:08:52.365000 2403842 site-packages/torch/_dynamo/utils.py:1915] [!0] ChromiumEventLogger: Start event not in stack, ignoring
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code] TRACED GRAPH
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]  ===== __compiled_fn_10_9e4ea995_3cda_4635_89d8_3a90a7ecffb4 =====
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]     def forward(self, L_inputs_ : list):
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         l_inputs_ = L_inputs_
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: <eval_with_key>.10:5 in forward, code: getitem = inputs[0]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem: "f32[][]cpu" = l_inputs_[0]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_1: "f32[1][1]cpu" = l_inputs_[1]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_2: "f32[8, 256][256, 1]cpu" = l_inputs_[2]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_3: "f32[2048][1]cpu" = l_inputs_[3]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_4: "f32[2048][1]cpu" = l_inputs_[4];  l_inputs_ = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: torch::autograd::GraphRoot (NodeCall 0)
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [], False)]);  getitem = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_5: "f32[][]cpu" = validate_outputs[0];  validate_outputs = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: CompiledFunctionBackward1 (NodeCall 1)
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         zeros: "f32[8, 256][256, 1]cpu" = torch.ops.aten.zeros.default([8, 256], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((getitem_1, getitem_2), [], getitem_5, zeros);  getitem_2 = getitem_5 = zeros = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_primals_1: "f32[1][1]cpu" = call_aot_bwd_prologue[0]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_primals_2: "f32[8, 256][256, 1]cpu" = call_aot_bwd_prologue[1]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_tangents_1: "f32[][]cpu" = call_aot_bwd_prologue[2]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_tangents_2: "f32[8, 256][256, 1]cpu" = call_aot_bwd_prologue[3];  call_aot_bwd_prologue = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(aot1_tangents_1, [8, 256]);  aot1_tangents_1 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:57 in train_step, code: loss = out.sum()
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(aot1_tangents_2, aot1_expand);  aot1_tangents_2 = aot1_expand = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(aot1_primals_1, aot1_primals_2)
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(aot1_mul);  aot1_mul = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(aot1_add, aot1_cos);  aot1_add = aot1_cos = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(aot1_mul_1, aot1_primals_1);  aot1_primals_1 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_mul_3: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(aot1_mul_1, aot1_primals_2);  aot1_mul_1 = aot1_primals_2 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(aot1_mul_3, [0, 1], True);  aot1_mul_3 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         aot1_view: "f32[1][1]cpu" = torch.ops.aten.reshape.default(aot1_sum_2, [1]);  aot1_sum_2 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/external_utils.py:226 in call_accumulate_grad, code: updated_grad = torch._dynamo.compiled_autograd.ops.AccumulateGrad(  # type: ignore[attr-defined]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot1_view], getitem_1, None, False);  aot1_view = getitem_1 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_10: "f32[1][1]cpu" = accumulate_grad[0];  accumulate_grad = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/compiled_autograd.py:1448 in set_node_origin, code: ViewBackward0 (NodeCall 2)
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         view_backward0 = torch__dynamo_compiled_autograd_ops_ViewBackward0([aot1_mul_2], [True], [2048]);  aot1_mul_2 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_11: "f32[2048][1]cpu" = view_backward0[0];  view_backward0 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         validate_outputs_1 = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem_11], [((None, None, device(type='cpu'), 6, 0, None), [2048], False)]);  getitem_11 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_12: "f32[2048][1]cpu" = validate_outputs_1[0];  validate_outputs_1 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/external_utils.py:226 in call_accumulate_grad, code: updated_grad = torch._dynamo.compiled_autograd.ops.AccumulateGrad(  # type: ignore[attr-defined]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([getitem_12], getitem_3, getitem_4, False);  getitem_12 = getitem_3 = getitem_4 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         getitem_13: "f32[2048][1]cpu" = accumulate_grad_1[0];  accumulate_grad_1 = None
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]         return (getitem_10, getitem_13)
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.427000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [!0/5/0] [__graph_code]
V0121 00:08:52.459000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:203] [!0/5/0] [__aot_graphs] aot_config id: 2, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=True, mutates_metadata=False, mutations_hidden_from_autograd=False, mutations_under_no_grad_or_inference_mode=True, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=False, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.is_input: 3>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=4, dynamic_dims=set(), requires_grad=False, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[], traced_tangents_descs=[], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None), PlainTensorMeta(unwrapped_idx=3, memory_format=None), PlainTensorMeta(unwrapped_idx=4, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[], is_train=False, traced_tangent_metas=None, num_symints_saved_for_bw=None, grad_enabled_mutation=None, deterministic=False, static_input_indices=[1], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=None, num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None),subclass_metadata=None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]  ===== Forward graph 2 =====
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class <lambda>(torch.nn.Module):
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]     def forward(self, arg0_1: "f32[][]cpu", arg1_1: "f32[1][1]cpu", arg2_1: "f32[8, 256][256, 1]cpu", arg3_1: "f32[2048][1]cpu", arg4_1: "f32[2048][1]cpu"):
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         # No stacktrace found for following nodes
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         full: "f32[8, 256][256, 1]cpu" = torch.ops.aten.full.default([8, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(arg0_1, [8, 256]);  arg0_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(full, expand);  full = expand = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1)
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(mul);  mul = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(add, cos);  add = cos = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, arg1_1);  arg1_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         mul_3: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         sum_1: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True);  mul_3 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         new_empty_strided: "f32[1][1]cpu" = torch.ops.aten.new_empty_strided.default(view, [1], [1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         copy: "f32[1][1]cpu" = torch.ops.aten.copy.default(new_empty_strided, view);  new_empty_strided = view = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         view_1: "f32[2048][1]cpu" = torch.ops.aten.view.default(mul_2, [2048]);  mul_2 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         add_1: "f32[2048][1]cpu" = torch.ops.aten.add.Tensor(arg4_1, view_1);  view_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         copy_: "f32[2048][1]cpu" = torch.ops.aten.copy_.default(arg4_1, add_1);  arg4_1 = add_1 = None
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]         return (copy, copy_)
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]
I0121 00:08:52.489000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [!0/5/0] [__aot_graphs]
W0121 00:08:53.499000 2403842 site-packages/torch/_logging/_internal.py:1199] [6/0] Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks] Graph break in user code at /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py:81
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks] Graph Break Reason: Call to `torch._dynamo.graph_break()`
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   Explanation: User-inserted graph break. Message: None
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   Hint: Remove the `torch._dynamo.graph_break()` call.
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks] User code traceback:
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 104, in <module>
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     sys.exit(main())
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 98, in main
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     compiled_train_step(Model(), data)
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 64, in compiled_train_step
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 59, in train_step
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     optimizer.step()
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py", line 517, in wrapper
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     out = func(*args, **kwargs)
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]   File "/opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py", line 81, in _use_grad
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]     torch._dynamo.graph_break()
V0121 00:08:53.508000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [6/0] [__graph_breaks]
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks] Graph break (user stack suppressed due to duplicate graph break) in user code at /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py:81
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks] Graph Break Reason: Call to `torch._dynamo.graph_break()`
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]   Explanation: User-inserted graph break. Message: None
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]   Hint: Remove the `torch._dynamo.graph_break()` call.
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]   Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]
V0121 00:08:53.536000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:618] [7/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code] TRACED GRAPH
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]  ===== __compiled_fn_19_2e467e4f_6ef6_44bf_810f_f628f4b28f09 =====
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]     def forward(self, L_self_param_groups_0_params_0_grad: "f32[1][1]cpu"):
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]         l_self_param_groups_0_params_0_grad = L_self_param_groups_0_params_0_grad
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:118 in step, code: for group in self.param_groups:
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]         p: "f32[1][1]cpu" = self.self___param_groups_0__params___0
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:375 in _single_tensor_sgd, code: param.add_(grad, alpha=-lr)
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]         mul: "f32[1][1]cpu" = torch.mul(l_self_param_groups_0_params_0_grad, -0.01);  l_self_param_groups_0_params_0_grad = None
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]         add_: "f32[1][1]cpu" = p.add_(mul);  p = mul = add_ = None
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]         return ()
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]
V0121 00:08:53.610000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [8/0] [__graph_code]
V0121 00:08:53.623000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:203] [8/0] [__aot_graphs] aot_config id: 3, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=True, mutates_metadata=False, mutations_hidden_from_autograd=False, mutations_under_no_grad_or_inference_mode=True, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True)], output_info=[], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[], traced_tangents_descs=[], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[], subclass_tangent_meta=[], is_train=False, traced_tangent_metas=None, num_symints_saved_for_bw=None, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=None, num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None),subclass_metadata=None
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]  ===== Forward graph 3 =====
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class <lambda>(torch.nn.Module):
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]     def forward(self, arg0_1: "f32[1][1]cpu", arg1_1: "f32[1][1]cpu"):
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:375 in _single_tensor_sgd, code: param.add_(grad, alpha=-lr)
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]         mul: "f32[1][1]cpu" = torch.ops.aten.mul.Tensor(arg1_1, -0.01);  arg1_1 = None
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]         add: "f32[1][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, mul);  mul = None
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]         copy_: "f32[1][1]cpu" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]         return ()
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]
I0121 00:08:53.634000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [8/0] [__aot_graphs]
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks] Graph break in user code at /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:60
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks] Graph Break Reason: Attempted to inline function marked as skipped
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   Explanation: Dynamo developers have intentionally marked that the function `Optimizer.zero_grad` should not be traced.
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   Hint: Avoid calling the function `Optimizer.zero_grad`.
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `Optimizer.zero_grad` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   Hint: Please file an issue to PyTorch.
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   Developer debug context: qualname: Optimizer.zero_grad, name: inner, filename: `/opt/conda/envs/tt/lib/python3.12/site-packages/torch/_compile.py`, skip reason: skipped according trace_rules.lookup MOD_SKIPLIST
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks] User code traceback:
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 104, in <module>
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]     sys.exit(main())
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 98, in main
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]     compiled_train_step(Model(), data)
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 64, in compiled_train_step
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 60, in train_step
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]     optimizer.zero_grad()
V0121 00:08:54.542000 2403842 site-packages/torch/_dynamo/symbolic_convert.py:611] [10/0] [__graph_breaks]





Style 4: torch.functional
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code] TRACED GRAPH
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]  ===== __compiled_fn_26_82c8a6c2_6106_4ffc_bb0e_6350cc9844e0 =====
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]     def forward(self, L_params_scale_: "f32[1][1]cpu", L_data_: "f32[8, 256][256, 1]cpu"):
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         l_params_scale_ = L_params_scale_
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         l_data_ = L_data_
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         # No stacktrace found for following nodes
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:110 in _wrap_tensor_for_grad, code: return _wrap_for_grad(maybe_tensor, level)
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         value: "f32[1][1]cpu" = torch._C._functorch._wrap_for_grad(l_params_scale_, 1);  l_params_scale_ = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _wrap_for_grad_1: "f32[8, 256][256, 1]cpu" = torch._C._functorch._wrap_for_grad(l_data_, 1);  l_data_ = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         # No stacktrace found for following nodes
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:77 in create_differentiable, code: return _set_tensor_requires_grad(x)
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _set_tensor_requires_grad: "f32[1][1]cpu" = torch._functorch.eager_transforms._set_tensor_requires_grad(value);  _set_tensor_requires_grad = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         # No stacktrace found for following nodes
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = value * _wrap_for_grad_1;  _wrap_for_grad_1 = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         sin: "f32[8, 256][256, 1]cpu" = mul.sin();  mul = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:73 in _func_model, code: return torch.func.functional_call(model, params, (data,)).sum()
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         output: "f32[][]cpu" = sin.sum();  sin = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [value], create_graph = True);  value = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         value_1: "f32[1][1]cpu" = _autograd_grad[0];  _autograd_grad = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:86 in unwrap_tensors, code: return _unwrap_for_grad(x, level)
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         value_2: "f32[1][1]cpu" = torch._C._functorch._unwrap_for_grad(value_1, 1);  value_1 = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:86 in unwrap_tensors, code: return _unwrap_for_grad(x, level)
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         output_1: "f32[][]cpu" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         # No stacktrace found for following nodes
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]         return (value_2,)
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
V0121 00:08:55.853000 2403842 site-packages/torch/_dynamo/output_graph.py:1983] [12/0] [__graph_code]
I0121 00:08:55.941000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [12/0] [__aot_graphs] aot_config id: 4, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.unsafe_view_alias: 7>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(1,))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.unsafe_view_alias: 7>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(1,))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]  ===== Forward graph 4 =====
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu"):
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         full: "f32[][]cpu" = torch.ops.aten.full.default([], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(full, [8, 256]);  full = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(mul);  mul = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_1, primals_2);  mul_1 = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True);  mul_2 = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]         return (view, primals_1, primals_2)
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]
I0121 00:08:55.942000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs] TRACED GRAPH
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]  ===== Backward graph 4 =====
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]  <eval_with_key>.18 class GraphModule(torch.nn.Module):
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", tangents_1: "f32[1][1]cpu"):
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         view_1: "f32[1, 1][1, 1]cpu" = torch.ops.aten.view.default(tangents_1, [1, 1]);  tangents_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         expand_1: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(view_1, [8, 256]);  view_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         full: "f32[][]cpu" = torch.ops.aten.full.default([], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(full, [8, 256]);  full = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         cos: "f32[8, 256][256, 1]cpu" = torch.ops.aten.cos.default(mul)
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand, cos);  cos = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_3: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand_1, mul_1);  mul_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_4: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand_1, primals_2);  expand_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_5: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_4, expand);  mul_4 = expand = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         sin_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.sin.default(mul);  mul = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         neg: "f32[8, 256][256, 1]cpu" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_6: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_5, neg);  mul_5 = neg = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_7: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_6, primals_1);  primals_1 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         mul_8: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(mul_6, primals_2);  mul_6 = primals_2 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         sum_3: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1], True);  mul_8 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         view_2: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:33 in forward, code: return (self.scale * x).sin()
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(mul_3, mul_7);  mul_3 = mul_7 = None
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]         return (view_2, add)
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
I0121 00:08:55.944000 2403842 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [12/0] [__aot_graphs]
```

"""

import sys
import time

import torch

torch._dynamo.config.compiled_autograd = True


class Model(torch.nn.Module):
    """Model with a single scale and single bias weight.

    Including a sine function makes it easier to distinguish
    fwd and bwd since sine's derivative is cosine.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return (self.scale * x).sin()


def eager(model, data):
    print("\n\n\n\n\nStyle 1: eager.", flush=True)
    out_eager = model(data)
    loss = out_eager.sum()
    loss.backward()


def compiled_module(model, data):
    print("\n\n\n\n\nStyle 2: torch.compile on the module.", flush=True)
    compiled_model = torch.compile(model)
    out_compiled_module = compiled_model(data)
    loss = out_compiled_module.sum()
    loss.backward()


def compiled_train_step(model, data):
    print("\n\n\n\n\nStyle 3: torch.compile on a compiled train_step().", flush=True)

    @torch.compile
    def train_step(model, data, optimizer):
        out = model(data)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return out

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _ = train_step(model, data, optimizer)


def compiled_functional(model, data):
    print("\n\n\n\n\nStyle 4: torch.functional", flush=True)

    @torch.compile
    def train_step(model, params, data):
        def _func_model(params, data):
            return torch.func.functional_call(model, params, (data,)).sum()

        grads, _ = torch.func.grad_and_value(_func_model)(params, data)
        for name, param in model.named_parameters():
            param.grad = grads[name]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _ = train_step(model, dict(model.named_parameters()), data)
    optimizer.step()


def main() -> int:
    """Runs multiple styles of compile and logs the lowerings.

    There is no criterion function (loss function) because the
    value is the loss.
    """
    B, F = 8, 256  # Batch size and feature size

    data = torch.arange(B * F, dtype=torch.float32, requires_grad=True).reshape(B, F)

    eager(Model(), data)
    time.sleep(1)
    compiled_module(Model(), data)
    time.sleep(1)
    compiled_train_step(Model(), data)
    time.sleep(1)
    compiled_functional(Model(), data)


if __name__ == "__main__":
    sys.exit(main())
