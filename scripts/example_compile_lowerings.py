"""Demonstrates lowerings for different styles of compile and torchfunctional.

Log output below, indicating the compiled train step DOES have a graph break
in the dynamo graph before lowering to the AOTAutograd graph.

However, the torch functional style does NOT have a graph break.

(tt) yho_google_com@yho-l4:~/Documents/GitHub/sandbox$ TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_LOGS="graph_code, graph_breaks, aot_graphs" python scripts/example_compile_lowerings.py
W0120 22:19:33.377000 2343082 site-packages/torch/_logging/_internal.py:474] Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs





Style 1: eager.





Style 2: torch.compile on the module.
/opt/conda/envs/tt/lib/python3.12/site-packages/torch/_dynamo/pgo.py:537: UserWarning: dynamo_pgo force disabled by torch.compiler.config.force_disable_caches
  warn_once(
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code] TRACED GRAPH
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]  ===== __compiled_fn_1_9b538145_9a94_4a7e_bb56_cdebf75a6257 =====
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]     def forward(self, L_self_parameters_scale_: "f32[1][1]cpu", L_x_: "f32[8, 256][256, 1]cpu", L_self_parameters_bias_: "f32[1][1]cpu"):
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         l_self_parameters_scale_ = L_self_parameters_scale_
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         l_x_ = L_x_
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         l_self_parameters_bias_ = L_self_parameters_bias_
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = l_self_parameters_scale_ * l_x_;  l_self_parameters_scale_ = l_x_ = None
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         add: "f32[8, 256][256, 1]cpu" = mul + l_self_parameters_bias_;  mul = l_self_parameters_bias_ = None
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]         return (add,)
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
V0120 22:19:37.029000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [0/0] [__graph_code]
I0120 22:19:37.445000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [0/0] [__aot_graphs] aot_config id: 0, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 2], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 2], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs] TRACED GRAPH
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]  ===== Forward graph 0 =====
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", primals_3: "f32[1][1]cpu"):
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(mul, primals_3);  mul = primals_3 = None
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]         return (add, primals_1, primals_2)
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]
I0120 22:19:37.446000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [0/0] [__aot_graphs]
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs] TRACED GRAPH
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]  ===== Backward graph 0 =====
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]  <eval_with_key>.2 class GraphModule(torch.nn.Module):
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", tangents_1: "f32[8, 256][256, 1]cpu"):
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         sum_1: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1], True)
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  primals_1 = None
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = primals_2 = None
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True);  mul_2 = None
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         view_1: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]         return (view_1, mul_1, view)
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]
I0120 22:19:37.447000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [0/0] [__aot_graphs]





Style 3: torch.compile on a compiled train_step().
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] Graph break in user code at /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:62
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] Graph Break Reason: Unsupported Tensor.backward() call
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Explanation: Dynamo currently does not support tracing `Tensor.backward()`.
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   Developer debug context: call_method TensorVariable() backward () {}
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0123.html
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks] User code traceback:
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 87, in <module>
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     sys.exit(main())
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 68, in main
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 62, in train_step
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]     loss.backward()
V0120 22:19:48.783000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [1/0] [__graph_breaks]
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code] TRACED GRAPH
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]  ===== __compiled_fn_4_f6192802_158d_42a8_882f_57ab74cc6de5 =====
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]     def forward(self, L_model_parameters_scale_: "f32[1][1]cpu", L_data_: "f32[8, 256][256, 1]cpu", L_model_parameters_bias_: "f32[1][1]cpu"):
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         l_model_parameters_scale_ = L_model_parameters_scale_
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         l_data_ = L_data_
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         l_model_parameters_bias_ = L_model_parameters_bias_
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = l_model_parameters_scale_ * l_data_;  l_model_parameters_scale_ = l_data_ = None
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         out: "f32[8, 256][256, 1]cpu" = mul + l_model_parameters_bias_;  mul = l_model_parameters_bias_ = None
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:61 in train_step, code: loss = out.sum()
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         loss: "f32[][]cpu" = out.sum()
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]         return (loss, out)
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
V0120 22:19:48.796000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [1/0_1] [__graph_code]
I0120 22:19:48.861000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [1/0_1] [__aot_graphs] aot_config id: 1, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=()), FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0)), TangentAOTInput(output=PlainAOTOutput(idx=1))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format)), PlainTensorMeta(unwrapped_idx=1, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 2], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=()), FakeTensor(..., size=(8, 256))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0)), TangentAOTInput(output=PlainAOTOutput(idx=1))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format)), PlainTensorMeta(unwrapped_idx=1, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 2], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs] TRACED GRAPH
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]  ===== Forward graph 1 =====
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", primals_3: "f32[1][1]cpu"):
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         mul: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         add: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(mul, primals_3);  mul = primals_3 = None
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:61 in train_step, code: loss = out.sum()
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         sum_1: "f32[][]cpu" = torch.ops.aten.sum.default(add)
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]         return (sum_1, add, primals_1, primals_2)
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0120 22:19:48.862000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [1/0_1] [__aot_graphs]
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs] TRACED GRAPH
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]  ===== Backward graph 1 =====
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]  <eval_with_key>.8 class GraphModule(torch.nn.Module):
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[8, 256][256, 1]cpu", tangents_1: "f32[][]cpu", tangents_2: "f32[8, 256][256, 1]cpu"):
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:61 in train_step, code: loss = out.sum()
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(tangents_1, [8, 256]);  tangents_1 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:61 in train_step, code: loss = out.sum()
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         add_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.add.Tensor(tangents_2, expand);  tangents_2 = expand = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1], True)
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(add_1, primals_1);  primals_1 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(add_1, primals_2);  add_1 = primals_2 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         sum_3: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True);  mul_2 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         view_1: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]         return (view_1, mul_1, view)
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
I0120 22:19:48.863000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [1/0_1] [__aot_graphs]
W0120 22:19:50.367000 2343082 site-packages/torch/_logging/_internal.py:1199] [3/0] Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks] Graph break in user code at /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py:81
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks] Graph Break Reason: Call to `torch._dynamo.graph_break()`
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   Explanation: User-inserted graph break. Message: None
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   Hint: Remove the `torch._dynamo.graph_break()` call.
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks] User code traceback:
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 87, in <module>
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]     sys.exit(main())
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 68, in main
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 63, in train_step
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]     optimizer.step()
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   File "/opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py", line 517, in wrapper
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]     out = func(*args, **kwargs)
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]   File "/opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py", line 81, in _use_grad
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]     torch._dynamo.graph_break()
V0120 22:19:50.385000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [3/0] [__graph_breaks]
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks] Graph break (user stack suppressed due to duplicate graph break) in user code at /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/optimizer.py:81
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks] Graph Break Reason: Call to `torch._dynamo.graph_break()`
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]   Explanation: User-inserted graph break. Message: None
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]   Hint: Remove the `torch._dynamo.graph_break()` call.
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]   Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]
V0120 22:19:50.415000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:618] [4/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code] TRACED GRAPH
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]  ===== __compiled_fn_15_b2f9e285_0352_4727_8618_43fe6cd329c2 =====
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]     def forward(self, L_self_param_groups_0_params_0_grad: "f32[1][1]cpu", L_self_param_groups_0_params_1_grad: "f32[1][1]cpu"):
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         l_self_param_groups_0_params_0_grad = L_self_param_groups_0_params_0_grad
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         l_self_param_groups_0_params_1_grad = L_self_param_groups_0_params_1_grad
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:118 in step, code: for group in self.param_groups:
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         p: "f32[1][1]cpu" = self.self___param_groups_0__params___0
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         param: "f32[1][1]cpu" = self.self___param_groups_0__params___1
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:375 in _single_tensor_sgd, code: param.add_(grad, alpha=-lr)
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         mul: "f32[1][1]cpu" = torch.mul(l_self_param_groups_0_params_0_grad, -0.01);  l_self_param_groups_0_params_0_grad = None
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         add_: "f32[1][1]cpu" = p.add_(mul);  p = mul = add_ = None
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         mul_1: "f32[1][1]cpu" = torch.mul(l_self_param_groups_0_params_1_grad, -0.01);  l_self_param_groups_0_params_1_grad = None
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         add__1: "f32[1][1]cpu" = param.add_(mul_1);  param = mul_1 = add__1 = None
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]         return ()
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]
V0120 22:19:50.497000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [5/0] [__graph_code]
V0120 22:19:50.513000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:203] [5/0] [__aot_graphs] aot_config id: 2, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=True, mutates_metadata=False, mutations_hidden_from_autograd=False, mutations_under_no_grad_or_inference_mode=True, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=True, mutates_metadata=False, mutations_hidden_from_autograd=False, mutations_under_no_grad_or_inference_mode=True, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=False, keep_input_mutations=True)], output_info=[], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[], traced_tangents_descs=[], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None), PlainTensorMeta(unwrapped_idx=3, memory_format=None)], subclass_fw_graph_out_meta=[], subclass_tangent_meta=[], is_train=False, traced_tangent_metas=None, num_symints_saved_for_bw=None, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 1], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=None, num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None),subclass_metadata=None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs] TRACED GRAPH
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]  ===== Forward graph 2 =====
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class <lambda>(torch.nn.Module):
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]     def forward(self, arg0_1: "f32[1][1]cpu", arg1_1: "f32[1][1]cpu", arg2_1: "f32[1][1]cpu", arg3_1: "f32[1][1]cpu"):
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/optim/sgd.py:375 in _single_tensor_sgd, code: param.add_(grad, alpha=-lr)
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         mul: "f32[1][1]cpu" = torch.ops.aten.mul.Tensor(arg2_1, -0.01);  arg2_1 = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         add: "f32[1][1]cpu" = torch.ops.aten.add.Tensor(arg0_1, mul);  mul = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         mul_1: "f32[1][1]cpu" = torch.ops.aten.mul.Tensor(arg3_1, -0.01);  arg3_1 = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         add_1: "f32[1][1]cpu" = torch.ops.aten.add.Tensor(arg1_1, mul_1);  mul_1 = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         copy_: "f32[1][1]cpu" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         copy__1: "f32[1][1]cpu" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = copy__1 = None
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]         return ()
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]
I0120 22:19:50.534000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_capture.py:289] [5/0] [__aot_graphs]
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks] Graph break in user code at /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:64
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks] Graph Break Reason: Attempted to inline function marked as skipped
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   Explanation: Dynamo developers have intentionally marked that the function `Optimizer.zero_grad` should not be traced.
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   Hint: Avoid calling the function `Optimizer.zero_grad`.
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `Optimizer.zero_grad` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   Hint: Please file an issue to PyTorch.
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   Developer debug context: qualname: Optimizer.zero_grad, name: inner, filename: `/opt/conda/envs/tt/lib/python3.12/site-packages/torch/_compile.py`, skip reason: skipped according trace_rules.lookup MOD_SKIPLIST
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks] User code traceback:
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 87, in <module>
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]     sys.exit(main())
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 68, in main
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]     _ = train_step(model, data, optimizer)
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]   File "/home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py", line 64, in train_step
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]     optimizer.zero_grad()
V0120 22:19:51.271000 2343082 site-packages/torch/_dynamo/symbolic_convert.py:611] [7/0] [__graph_breaks]





Style 4: torch.functional
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code] TRACED GRAPH
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]  ===== __compiled_fn_22_2353ee3f_4aa0_4098_94db_7c6abd942ba4 =====
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]     def forward(self, L_data_: "f32[8, 256][256, 1]cpu"):
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         l_data_ = L_data_
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         # No stacktrace found for following nodes
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/utils/_pytree.py:1282 in helper, code: subspecs = [helper(child, leaves) for child in children]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         child: "f32[1][1]cpu" = self.params____scale
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         child_1: "f32[1][1]cpu" = self.params____bias
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:110 in _wrap_tensor_for_grad, code: return _wrap_for_grad(maybe_tensor, level)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value: "f32[1][1]cpu" = torch._C._functorch._wrap_for_grad(child, 1);  child = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value_1: "f32[1][1]cpu" = torch._C._functorch._wrap_for_grad(child_1, 1);  child_1 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _wrap_for_grad_2: "f32[8, 256][256, 1]cpu" = torch._C._functorch._wrap_for_grad(l_data_, 1);  l_data_ = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         # No stacktrace found for following nodes
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:77 in create_differentiable, code: return _set_tensor_requires_grad(x)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _set_tensor_requires_grad: "f32[1][1]cpu" = torch._functorch.eager_transforms._set_tensor_requires_grad(value);  _set_tensor_requires_grad = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         # No stacktrace found for following nodes
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:77 in create_differentiable, code: return _set_tensor_requires_grad(x)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _set_tensor_requires_grad_1: "f32[1][1]cpu" = torch._functorch.eager_transforms._set_tensor_requires_grad(value_1);  _set_tensor_requires_grad_1 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         # No stacktrace found for following nodes
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:28 in forward, code: return self.scale * x + self.bias
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         mul: "f32[8, 256][256, 1]cpu" = value * _wrap_for_grad_2;  _wrap_for_grad_2 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         add: "f32[8, 256][256, 1]cpu" = mul + value_1;  mul = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /home/yho_google_com/Documents/GitHub/sandbox/scripts/example_compile_lowerings.py:75 in _func_model, code: return torch.func.functional_call(model, params, (data,)).sum()
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         output: "f32[][]cpu" = add.sum();  add = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [value, value_1], create_graph = True);  value = value_1 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value_2: "f32[1][1]cpu" = _autograd_grad[0]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value_3: "f32[1][1]cpu" = _autograd_grad[1];  _autograd_grad = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:86 in unwrap_tensors, code: return _unwrap_for_grad(x, level)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value_4: "f32[1][1]cpu" = torch._C._functorch._unwrap_for_grad(value_2, 1);  value_2 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         value_5: "f32[1][1]cpu" = torch._C._functorch._unwrap_for_grad(value_3, 1);  value_3 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:86 in unwrap_tensors, code: return _unwrap_for_grad(x, level)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         output_1: "f32[][]cpu" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         # No stacktrace found for following nodes
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]         return (value_4, value_5)
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
V0120 22:19:51.776000 2343082 site-packages/torch/_dynamo/output_graph.py:1983] [9/0] [__graph_code]
I0120 22:19:51.845000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1475] [9/0] [__aot_graphs] aot_config id: 3, fw_metadata=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.unsafe_view_alias: 7>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=False, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(1,))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 1], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None), inner_meta=ViewAndMutationMeta(input_info=[InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=True, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True), InputAliasInfo(is_leaf=False, mutates_data=False, mutates_metadata=False, mutations_hidden_from_autograd=True, mutations_under_no_grad_or_inference_mode=False, mutation_inductor_storage_resize=False, mutates_storage_metadata=False, requires_grad=True, keep_input_mutations=True)], output_info=[OutputAliasInfo(output_type=<OutputType.unsafe_view_alias: 7>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=True, view_meta_sequence=None), OutputAliasInfo(output_type=<OutputType.non_alias: 1>, raw_type=<class 'torch._subclasses.functional_tensor.FunctionalTensor'>, base_idx=None, dynamic_dims=set(), requires_grad=False, view_meta_sequence=None)], num_intermediate_bases=0, keep_input_mutations=True, traced_tangents=[FakeTensor(..., size=(1,))], traced_tangents_descs=[TangentAOTInput(output=PlainAOTOutput(idx=0))], subclass_inp_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None), PlainTensorMeta(unwrapped_idx=2, memory_format=None)], subclass_fw_graph_out_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=None), PlainTensorMeta(unwrapped_idx=1, memory_format=None)], subclass_tangent_meta=[PlainTensorMeta(unwrapped_idx=0, memory_format=MemoryFormatMeta(size=None, stride=None, memory_format=torch.contiguous_format))], is_train=True, traced_tangent_metas=None, num_symints_saved_for_bw=0, grad_enabled_mutation=None, deterministic=False, static_input_indices=[0, 1], tokens={}, indices_of_inputs_that_requires_grad_with_mutations_in_bw=[], bw_donated_idxs=[], num_backward_tokens=0, num_graphsafe_rng_states=0, graphsafe_rng_state_index=None)
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs] TRACED GRAPH
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]  ===== Forward graph 3 =====
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]  /opt/conda/envs/tt/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]     def forward(self, primals_1: "f32[1][1]cpu", primals_2: "f32[1][1]cpu", primals_3: "f32[8, 256][256, 1]cpu"):
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         full: "f32[][]cpu" = torch.ops.aten.full.default([], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(full, [8, 256]);  full = None
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         sum_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(expand, [0, 1], True)
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         view: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         mul_1: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand, primals_3);  expand = primals_3 = None
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         sum_3: "f32[1, 1][1, 1]cpu" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1], True);  mul_1 = None
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         view_1: "f32[1][1]cpu" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]         return (view_1, view)
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]
I0120 22:19:51.846000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1581] [9/0] [__aot_graphs]
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs] TRACED GRAPH
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]  ===== Backward graph 3 =====
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]  <eval_with_key>.16 class GraphModule(torch.nn.Module):
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]     def forward(self, tangents_1: "f32[1][1]cpu"):
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]          # File: /opt/conda/envs/tt/lib/python3.12/site-packages/torch/_functorch/eager_transforms.py:1390 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         view_2: "f32[1, 1][1, 1]cpu" = torch.ops.aten.view.default(tangents_1, [1, 1]);  tangents_1 = None
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         expand_1: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(view_2, [8, 256]);  view_2 = None
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         full: "f32[][]cpu" = torch.ops.aten.full.default([], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         expand: "f32[8, 256][0, 0]cpu" = torch.ops.aten.expand.default(full, [8, 256]);  full = None
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         mul_2: "f32[8, 256][256, 1]cpu" = torch.ops.aten.mul.Tensor(expand_1, expand);  expand_1 = expand = None
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]         return (None, None, mul_2)
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]
I0120 22:19:51.847000 2343082 site-packages/torch/_functorch/_aot_autograd/graph_compile.py:1592] [9/0] [__aot_graphs]

"""

import sys
import time

import torch

torch._logging.set_logs(aot_graphs=True)


class Model(torch.nn.Module):
    """Model with a single scale and single bias weight."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.scale * x + self.bias


def main() -> int:
    """Runs multiple styles of compile and logs the lowerings.

    There is no criterion function (loss function) because the
    value is the loss.
    """
    B, F = 8, 256  # Batch size and feature size

    data = torch.arange(B * F, dtype=torch.float32, requires_grad=True).reshape(B, F)
    model = Model()

    # Style 1 is eager. There is no logging output for this style.
    print("\n\n\n\n\nStyle 1: eager.", flush=True)
    out_eager = model(data)
    loss = out_eager.sum()
    loss.backward()
    time.sleep(1)

    print("\n\n\n\n\nStyle 2: torch.compile on the module.", flush=True)
    compiled_model = torch.compile(model)
    out_compiled_module = compiled_model(data)
    loss = out_compiled_module.sum()
    loss.backward()
    time.sleep(1)

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

    print("\n\n\n\n\nStyle 4: torch.functional", flush=True)

    @torch.compile
    def train_step(model, params, data, optimizer):
        def _func_model(params, data):
            return torch.func.functional_call(model, params, (data,)).sum()

        grads, _ = torch.func.grad_and_value(_func_model)(params, data)
        for name, param in model.named_parameters():
            param.grad = grads[name]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    _ = train_step(model, dict(model.named_parameters()), data, optimizer)
    optimizer.step()


if __name__ == "__main__":
    sys.exit(main())
