@group(0) @binding(0)
var<uniform> globals: GlobalData;
@group(0) @binding(1)
var<storage, read> previous_layer: array<MainType>;
@group(0) @binding(2)
var<storage, read> layer_bias: array<MainType>;
@group(0) @binding(3)
var<storage, read> layer_activation: array<MainType>;
@group(0) @binding(4)
var<storage, read_write> layer_output: array<MainType>;

@compute @workgroup_size(64)
fn compute_forwards(
  @builtin(global_invocation_id)
  global_id : vec3u
) {
    if (global_id.x >= globals.output_size) {
        return;
    }
    
    var output = MainType(0);

    for (var i = u32(0); i < globals.input_size; i++) {
        let matrix_index = i + globals.input_size * global_id.x;
        output += (previous_layer[i] * layer_activation[matrix_index]) + layer_bias[matrix_index];
    }
    layer_output[global_id.x] = logistic_function(output);
}