use std::str::FromStr;

use color_processing::Color as LibColor;

use crate::layer::MainType;

/// Represents a colour in the oklab colour space
#[derive(Debug)]
pub struct Color {
    pub l: MainType,
    pub a: MainType,
    pub b: MainType,
}

impl From<&[MainType]> for Color {
    fn from(value: &[MainType]) -> Self {
        Self {
            l: value[0],
            a: value[1],
            b: value[2],
        }
    }
}

impl Color {
    pub fn from_str(str: &str) -> Result<Self, <LibColor as FromStr>::Err> {
        let rgb = LibColor::from_str(str)?;
        let rgb = (
            rgb.red as MainType / 255 as MainType,
            rgb.green as MainType / 255 as MainType,
            rgb.blue as MainType / 255 as MainType
        );
        
        let linear_srgb = srgb_to_linear_srgb(rgb);
        let oklab = Self::from_linear_srgb(linear_srgb);
        return Ok(oklab);
    }

    pub fn from_linear_srgb(rgb: (MainType, MainType, MainType)) -> Self {
        let l = 0.4122214708 * rgb.0 + 0.5363325363 * rgb.1 + 0.0514459929 * rgb.2;
        let m = 0.2119034982 * rgb.0 + 0.6806995451 * rgb.1 + 0.1073969566 * rgb.2;
        let s = 0.0883024619 * rgb.0 + 0.2817188376 * rgb.1 + 0.6299787005 * rgb.2;

        let l = l.cbrt();
        let m = m.cbrt();
        let s = s.cbrt();

        return Self {
            l: 0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            a: 1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            b: 0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
        };
    }
}

fn srgb_to_linear_srgb(rgb: (MainType, MainType, MainType)) -> (MainType, MainType, MainType) {
    (f_inv(rgb.0), f_inv(rgb.1), f_inv(rgb.2))
}

fn f_inv(x: MainType)  -> MainType {
    return if x >= 0.04045 {
        ((x + 0.055)/(1.0 + 0.055)).powf(2.4)
    } else {
        x / 12.92
    };
}