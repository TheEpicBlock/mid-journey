alias MainType = f32;

struct GlobalData {
  input_size: u32,
  output_size: u32
}

fn logistic_function(x: MainType) -> MainType {
  return MainType(1) / (MainType(1) + exp(-x));
}
