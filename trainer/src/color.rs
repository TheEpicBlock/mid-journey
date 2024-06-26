use std::str::FromStr;

use bytemuck::{Pod, Zeroable};
use color_processing::Color as LibColor;

use crate::layer::MainType;

/// Represents a colour in the oklab colour space
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
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
        
        let oklab = Self::from_rgb(rgb);
        return Ok(oklab);
    }

    pub fn from_rgb(rgb: (MainType, MainType, MainType)) -> Self {
        return Self::from_linear_srgb(srgb_to_linear_srgb(rgb));
    }
    
    pub fn from_linear_srgb(rgb: (MainType, MainType, MainType)) -> Self {
        let l = 0.4122214708 * rgb.0 + 0.5363325363 * rgb.1 + 0.0514459929 * rgb.2;
        let m = 0.2119034982 * rgb.0 + 0.6806995451 * rgb.1 + 0.1073969566 * rgb.2;
        let s = 0.0883024619 * rgb.0 + 0.2817188376 * rgb.1 + 0.6299787005 * rgb.2;

        let l = l.cbrt();
        let m = m.cbrt();
        let s = s.cbrt();

        return Self::from_oklab((
            0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
        ));
    }

    pub fn from_oklab(oklab: (MainType, MainType, MainType)) -> Self {
        Self {
            l: oklab.0,
            a: (oklab.1 + 0.4) / 0.8,
            b: (oklab.2 + 0.4) / 0.8
        }
    }

    pub fn to_oklab(&self) -> (MainType, MainType, MainType) {
        return (
            self.l,
            (self.a * 0.8) - 0.4,
            (self.b * 0.8) - 0.4,
        );
    }

    pub fn to_linear_srgb(&self) -> (MainType, MainType, MainType) {
        let oklab = self.to_oklab();

        let l = oklab.0 + 0.3963377774 * oklab.1 + 0.2158037573 * oklab.2;
        let m = oklab.0 - 0.1055613458 * oklab.1 - 0.0638541728 * oklab.2;
        let s = oklab.0 - 0.0894841775 * oklab.1 - 1.2914855480 * oklab.2;

        let l = l*l*l;
        let m = m*m*m;
        let s = s*s*s;

        return (
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        );
    }

    pub fn to_rgb(&self) -> (MainType, MainType, MainType) {
        let linear_srgb = self.to_linear_srgb();
        return (f(linear_srgb.0), f(linear_srgb.1), f(linear_srgb.2));
    }

    pub fn to_hex(&self) -> String {
        let rgb = self.to_rgb();
        LibColor::new_rgb(
            (rgb.0 * 255.0) as u8,
            (rgb.1 * 255.0) as u8,
            (rgb.2 * 255.0) as u8
        ).to_hex_string()
    }
}

fn srgb_to_linear_srgb(rgb: (MainType, MainType, MainType)) -> (MainType, MainType, MainType) {
    (f_inv(rgb.0), f_inv(rgb.1), f_inv(rgb.2))
}

fn f_inv(x: MainType)  -> MainType {
    if x >= 0.04045 {
        ((x + 0.055)/(1.0 + 0.055)).powf(2.4)
    } else {
        x / 12.92
    }
}

fn f(x: MainType) -> MainType {
    if x >= 0.0031308 {
        (1.055) * x.powf(1.0/2.4) - 0.055
    } else {
        12.92 * x
    }
}

#[cfg(test)]
mod test {
    use crate::{color::Color, layer::MainType};

    #[test]
    fn test_oklab_conversion() {
        test((1.0, 1.0, 1.0), (1.0, 0.0, 0.0));
        test((1.0, 0.0, 0.0), (0.628, 0.225, 0.126));
        test((0.0, 1.0, 0.0), (0.866, -0.234, 0.179));
        test((0.0, 0.0, 1.0), (0.452, -0.032, -0.312));
        test((0.91, 0.21, 0.0), (0.608, 0.182, 0.122));
    }

    fn test(rgb: (MainType, MainType, MainType), oklab: (MainType, MainType, MainType)) {
        let oklab = Color::from_oklab(oklab);

        let oklab_converted = Color::from_rgb(rgb);
        assert!(cdist(oklab_converted, oklab) < 0.002, "Expected {oklab:?}, found {oklab_converted:?}");

        let rgb_converted = oklab.to_rgb();
        assert!(dist(rgb_converted, rgb) < 0.03, "Expected {rgb:?}, found {rgb_converted:?}");

        let round_trip_rgb = Color::from_rgb(rgb).to_rgb();
        assert!(dist(rgb, round_trip_rgb) < 0.0001, "Roundtripping rgb {rgb:?} should result in the same value but gave {round_trip_rgb:?} instead");

        let round_trip_oklab = Color::from_rgb(oklab.to_rgb());
        assert!(cdist(oklab, round_trip_oklab) < 0.0000005, "Roundtripping oklab {oklab:?} should result in the same value but gave {round_trip_oklab:?} instead");
    }

    fn cdist(a: Color, b: Color) -> MainType {
        dist((a.l, a.a, a.b), (b.l, b.a, b.b))
    }

    fn dist(a: (MainType, MainType, MainType), b: (MainType, MainType, MainType)) -> MainType {
        return ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
    }
}