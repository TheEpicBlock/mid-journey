/*
 * The backpropagation(_start) shaders have computed arrays of derivatives for the biases of each layer.
 * For each node, the bias derivative needs to be averaged across all iterations. Then, the we sum the
 * averaged derivative with the original bias to compute the new bias
 */

// The amount of nodes of the previous layer
override previous_layer_size: u32;
// The amount of nodes for this layer
override layer_size: u32;
// The number of invocations of biases that need to be averaged
override invocations: u32;
// Turns out naga doesn't like pipeline overridable constants being used like this one is
// That's why I'm doing an at-home version
const workers_per_node: u32 = ${workers_per_node};

// The activations of the previous layer
// type: array<array<MainType, previous_layer_size>, invocations>
@group(0) @binding(0)
var<storage, read> previous_layer_a: array<MainType>;
// The derivatives of the z function for each node in the layer that the weights connect to.
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(1)
var<storage, read> next_derivZ: array<MainType>;
// The weights for each connection between the previous and next layer
// type: array<array<MainType, previous_layer_size>, layer_size>
@group(0) @binding(2)
var<storage, read_write> weights: array<MainType>;

// Should match constants in `BackpropApplyWeightShaderPipeline` in shaders/mod.rs
const WORKGROUP_SIZE_A: u32 = 8;
const WORKGROUP_SIZE_B: u32 = 1;

var<workgroup> tempstorage_sum: array<array<array<MainType, workers_per_node>, WORKGROUP_SIZE_A>, WORKGROUP_SIZE_B>;

@compute @workgroup_size(workers_per_node, WORKGROUP_SIZE_A, WORKGROUP_SIZE_B)
fn apply_weights(
  @builtin(global_invocation_id)
  global_id: vec3u,
  @builtin(local_invocation_id)
  local_id: vec3u
) {
    // global_id.y will be the node of the previous layer
    if (global_id.y >= previous_layer_size) {
        return;
    }
    // global_id.z will be the node of the next layer
    if (global_id.z >= layer_size) {
        return;
    }

    // Each node has a number of workers which will sum the derivatives in parallel
    // which worker we are is indicated by local_id.yx
    var iters = invocations / workers_per_node;
    if (local_id.x == workers_per_node - 1) {
        iters += invocations % workers_per_node; // One worker needs to handle the remainder
    }
    
    tempstorage_sum[local_id.z][local_id.y][local_id.x] = MainType(0);
    for (var i: u32 = 0; i < iters; i++) {
        let invocation_index = (local_id.x * iters + i);
        // Unlike the bias, which is already computed, we need to do a little extra multiplication
        // to get the derivative of the weight
        // Like explained in math.md, derivW is equal to the activation times derivZ
        let derivative = previous_layer_a[global_id.y + invocation_index * previous_layer_size] * next_derivZ[global_id.z + invocation_index * layer_size];
        // Add it to the sum so we can average out the iterations.
        tempstorage_sum[local_id.z][local_id.y][local_id.x] += derivative;
    }

    // Wait until everyone is done
    workgroupBarrier();

    // Now we make the first worker sum up everything and write the result
    if (local_id.x == 0) {
        var sum: MainType = 0;
        for (var i: u32 = 0; i < workers_per_node; i++) {
            sum += tempstorage_sum[local_id.z][local_id.y][i];
        }
        weights[global_id.y + global_id.z * previous_layer_size] -= (learning_rate * sum) / MainType(invocations);
    }
}