package nl.theepicblock.mid.journey;

import net.minecraft.util.math.ColorHelper;

public class OkLab {
    public static int networkToMc(float[] input) {
        // The network uses scaled oklab values

        // Convert scaled to regular oklab
        var l = input[0];
        var a = (input[1] * 0.8) - 0.4;
        var b = (input[2] * 0.8) - 0.4;

        // Convert to linear rgb
        var _l = l + 0.3963377774 * a + 0.2158037573 * b;
        var _m = l - 0.1055613458 * a - 0.0638541728 * b;
        var _s = l - 0.0894841775 * a - 1.2914855480 * b;

        _l = _l*_l*_l;
        _m = _m*_m*_m;
        _s = _s*_s*_s;

        var _r = 4.0767416621 * _l - 3.3077115913 * _m + 0.2309699292 * _s;
        var _g = -1.2684380046 * _l + 2.6097574011 * _m - 0.3413193965 * _s;
        var _b = -0.0041960863 * _l - 0.7034186147 * _m + 1.7076147010 * _s;

        // Convert to srgb
        var __r = (float)f(_r);
        var __g = (float)f(_g);
        var __b = (float)f(_b);

        return ColorHelper.Argb.fromFloats(1.0f, __r, __g, __b);
    }

    private static double f(double x) {
        if (x >= 0.0031308) {
            return 1.055 * Math.pow(x, 1.0/2.4) - 0.055;
        } else {
            return 12.92 * x;
        }
    }
}
