use crate::layer::MainType;

#[derive(Debug)]
pub struct Color {
    r: MainType,
    g: MainType,
    b: MainType,
}

impl From<&[MainType]> for Color {
    fn from(value: &[MainType]) -> Self {
        Self {
            r: value[0],
            g: value[1],
            b: value[2],
        }
    }
}