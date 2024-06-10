/*
 * The backpropagation(_start) shaders have computed arrays of derivatives for the biases of each layer.
 * For each node, the bias derivative needs to be averaged across all iterations. Then, the we sum the
 * averaged derivative with the original bias to compute the new bias
 */

// The amount of nodes for this layer
override layer_size: u32;
// The number of invocations of biases that need to be averaged
override invocations: u32;
// Turns out naga doesn't like pipeline overridable constants being used like this one is
// That's why I'm doing an at-home version
const workers_per_node: u32 = ${workers_per_node};

// The derivatives of the z function for each node in this layer.
// This is equal to the derivative of the bias of each node
// type: array<array<MainType, layer_size>, invocations>
@group(0) @binding(0)
var<storage, read> derivZ: array<MainType>;
// The biases of the network for this layer.
// This is the same across all invocations
// type: array<MainType, output_size>
@group(0) @binding(1)
var<storage, read_write> biases: array<MainType>;

const WORKGROUP_SIZE: u32 = 16;

// type: array<array<MainType, workers_per_node>, layer_size>
var<workgroup> tempstorage_sum: array<array<MainType, workers_per_node>, WORKGROUP_SIZE>;

@compute @workgroup_size(workers_per_node, WORKGROUP_SIZE)
fn apply_biases(
  @builtin(global_invocation_id)
  global_id: vec3u,
  @builtin(local_invocation_id)
  local_id: vec3u
) {
    // global_id.y will be the node we're working on
    if (global_id.y >= layer_size) {
        return;
    }

    // Each node has a number of workers which will sum the derivatives in parallel
    // which worker we are is indicated by local_id.y
    var iters = invocations / workers_per_node;
    if (local_id.y == workers_per_node - 1) {
        iters += invocations % workers_per_node; // One worker needs to handle the remainder
    }
    for (var i: u32 = 0; i < iters; i++) {
        tempstorage_sum[local_id.y][local_id.x] += derivZ[global_id.y + (local_id.x * iters + i) * layer_size];
    }

    // Wait until everyone is done
    workgroupBarrier();

    // Now we make the first worker sum up everything and write the result
    if (local_id.x == 0) {
        var sum: MainType = 0;
        for (var i: u32 = 0; i < workers_per_node; i++) {
            sum += tempstorage_sum[local_id.y][i];
        }
        biases[global_id.y] += sum / MainType(invocations);
    }
}