alias MainType = f32;

// Should match the constants in shaders/mod.rs
const STD_WORKGROUP_SIZE = vec3(32, 2, 1);

const learning_rate: MainType = 0.01;

// The activation function
fn activation(x: MainType) -> MainType {
    if (x >= 0.0) {
        return x;
    } else {
        return 0.01 * x;
    }
}

// The derivative of the activation function
fn dActivation(x: MainType) -> MainType {
    if (x >= 0.0) {
        return 1.0;
    } else {
        return 0.01;
    }
}