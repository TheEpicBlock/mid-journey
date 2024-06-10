// I sometimes use derivX here to mean "derivative of C₀ with respect to X"

// The amount of nodes in the next layer
// Backpropagation works backwards, so 
// we actually use the next layer as input to compute the current layer.
// Eg: we use layer L to compute L-1, unlike `compute_forwards` which uses L-1 to compute L
override next_layer_size: u32;
// The amount of nodes in the input layer
override layer_size: u32;
// The number of invocations that this shader will do at once
override invocations: u32;

// The weights connecting this layer to the next layer
// type: array<array<MainType, layer_size>, next_layer_size>
@group(0) @binding(0)
var<storage, read> next_layer_weights: array<MainType>;
// The z-values of the layer
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(1)
var<storage, read> layer_z: array<MainType>;
// The derivatives of Z of the next layer. Like mentioned,
// these are used as inputs, not outputs.
// type: array<array<MainType, next_layer_size>, invocations>
@group(0) @binding(2)
var<storage, read> next_layer_derivZ: array<MainType>;

// The derivatives of the z function for each node in this layer.
// This is equal to the derivative of the bias of each node
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(3)
var<storage, read_write> derivZ: array<MainType>;

// This shader is used for all backpropagation steps except the first.
// It derives (dC_0/dz) from the (dC_0/dz) values of the previous step
// See math.md
@compute @workgroup_size(STD_WORKGROUP_SIZE.x, STD_WORKGROUP_SIZE.y, STD_WORKGROUP_SIZE.z)
fn backprop_from_layer(
  @builtin(global_invocation_id)
  global_id: vec3u
) {
    if (is_out_of_bounds(global_id)) {
        return;
    }

    var derivA: MainType = 0;
    
    // Using j as variable here to match the sum in the math.md equation
    // the `i` in math.md matches `global_id.y` here
    for (var j: u32 = 0; j < next_layer_size; j++) {
        let weight = next_layer_weights[global_id.y + j * layer_size];
        let next_derivZ = next_layer_derivZ[j + global_id.x * next_layer_size];
        derivA += weight * next_derivZ;
    }

    let i = global_id.y + global_id.x * layer_size;
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