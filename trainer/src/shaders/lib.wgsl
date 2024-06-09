alias MainType = f32;

// Should match the constants in shaders/mod.rs
const workgroup_size = vec3(32, 2, 1);

// The activation function
fn activation(x: MainType) -> MainType {
    return logistic_function(x);
}

// The derivative of the activation function
fn dActivation(x: MainType) -> MainType {
    // See https://en.wikipedia.org/wiki/Logistic_function#Derivative
    let l = logistic_function(x);
    return l * (1 - l);
}

fn logistic_function(x: MainType) -> MainType {
    return MainType(1) / (MainType(1) + exp(-x));
}