alias MainType = f32;

fn logistic_function(x: MainType) -> MainType {
  return MainType(1) / (MainType(1) + exp(-x));
}
