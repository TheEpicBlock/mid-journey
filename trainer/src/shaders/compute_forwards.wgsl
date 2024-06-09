// The amount of nodes in the input layer
override input_size: u32;
// The amount of nodes in the output layer
override output_size: u32;
// The number of invocations that this shader will do at once
override invocations: u32;

// The weights for each connection between the input and output layer
// type: array<array<MainType, input_size>, output_size>
@group(0) @binding(0)
var<storage, read> weights: array<MainType>;
// The biases of the network for this layer.
// This is the same across all invocations
// type: array<MainType, output_size>
@group(0) @binding(1)
var<storage, read> biases: array<MainType>;

// The activations of the previous layer
// type: array<array<MainType, input_size>, invocations>
@group(0) @binding(2)
var<storage, read> input_a: array<MainType>;
// The output "z" values for this layer. See math.md
// type: array<array<MainType, output_size>, invocations>
@group(0) @binding(3)
var<storage, read_write> output_z: array<MainType>;
// The output "a" values for this layer. See math.md
// type: array<array<MainType, output_size>, invocations>
@group(0) @binding(4)
var<storage, read_write> output_a: array<MainType>;

@compute @workgroup_size(64)
fn compute_forwards(
  @builtin(global_invocation_id)
  global_id : vec3u
) {
    // global_id.x represents which invocation we're in
    // (this shader is meant to run the same neural network on multiple inputs at once)
    if (global_id.x >= invocations) {
        return;
    }
    // global_id.y is the index of the node in the output layer which we want to compute
    if (global_id.y >= output_size) {
        return;
    }

    var output: MainType = 0;
    
    for (var i: u32 = 0; i < input_size; i++) {
        // The weight from the node in the previous layer to the node we're calculating
        let weight = weights[global_id.y + i * input_size];
        let input_activation = input_a[i + global_id.x * invocations];
        output += input_activation * weight;
    }

    output_a[global_id.y + global_id.x * invocations] = output;
    output_z[global_id.y + global_id.x * invocations] = logistic_function(output);
}