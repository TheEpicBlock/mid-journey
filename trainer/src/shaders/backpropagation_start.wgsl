// I sometimes use derivX here to mean "derivative of C₀ with respect to X"

// The amount of nodes in the input layer
override layer_size: u32;
// The number of invocations that this shader will do at once
override invocations: u32;

// The activations of the layer
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(0)
var<storage, read> layer_a: array<MainType>;
// The z-values of the layer
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(1)
var<storage, read> layer_z: array<MainType>;
// The expected activations of the layer
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(2)
var<storage, read> expected_a: array<MainType>;

// The derivatives of the z function for each node in this layer.
// This is eqal to the derivative of the bias of each node
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(3)
var<storage, read_write> derivZ: array<MainType>;

// This shader is used for the first backpropagation step.
// It derives (dC_0/dz) directly from the cost function
// See math.md 
@compute @workgroup_size(workgroup_size.x, workgroup_size.y, workgroup_size.z)
fn backprop_from_cost(
  @builtin(global_invocation_id)
  global_id: vec3u
) {
    if (is_out_of_bounds(global_id)) {
        return;
    }

    let i = global_id.y + global_id.x * layer_size;
    
    // Formula explained in math.md
    let derivA = 2*(layer_a[i] - expected_a[i]);
    
    derivZ[i] = dActivation(layer_z[i]) * derivA;
}

fn is_out_of_bounds(global_id: vec3u) -> bool {
    // global_id.x represents which invocation we're in
    // (this shader is meant to run the same neural network on multiple inputs at once)
    if (global_id.x >= invocations) {
        return true;
    }
    // global_id.y is the index of the node in the output layer which we want to compute
    if (global_id.y >= layer_size) {
        return true;
    }
    return false;
}