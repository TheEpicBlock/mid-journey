package nl.theepicblock.mid.journey.nn;

import com.google.common.collect.Iterables;
import joptsimple.internal.Strings;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import nl.theepicblock.mid.journey.MidJourneyClient;
import nl.theepicblock.mid.journey.OkLab;

import java.util.Arrays;
import java.util.Objects;

public class NeuralNetwork {
    @Environment(EnvType.CLIENT)
    public static int eval(String input) {
        return eval(input, MidJourneyClient.NN_CONFIG, MidJourneyClient.NN_PARAMETERS);
    }

    public static int eval(String input, NNConfig config, NetworkParameters[] parameters) {
        float[] previousLayer = createFirstLayer(input, config);
        float[] nextLayer;

        for (var layer = 0; layer < config.layers().length; layer++) {
            var layerSize = config.layers()[layer];
            var layerData = parameters[layer];

            nextLayer = new float[layerSize];
            for (int next = 0; next < nextLayer.length; next++) {
                float tmp = 0;
                for (int prev = 0; prev < previousLayer.length; prev++) {
                    tmp += previousLayer[prev] * layerData.weights()[prev + next * previousLayer.length];
                }
                tmp += layerData.biases()[next];
                nextLayer[next] = activationFunction(tmp);
            }
            previousLayer = nextLayer;
        }

        return OkLab.networkToMc(previousLayer);
    }

    private static float activationFunction(float in) {
        if (in > 0) {
            return in;
        } else {
            return 0.01f * in;
        }
    }

    private static float[] createFirstLayer(String input, NNConfig config) {
        // This logic MUST match the one in trainer/src/string.rs
        var output = new float[config.inputLength() * 27];

        var words = input.split("\\s");
        var lastWordI = -1;
        for (int i = 0; i < words.length; i++) {
            if (!words[i].startsWith("(")) {
                lastWordI = i;
            }
        }
        var lastWord = words[lastWordI];
        words[lastWordI] = null;

        var chars = Strings.join(Iterables.filter(Arrays.asList(words), Objects::nonNull), " ").chars().iterator();
        int i = 0;
        while (chars.hasNext()) {
            var n = charToNum(chars.next());
            if (n != -1) {
                output[i * 27 + n] = 1f;
            }
            i++;
        }

        chars = lastWord.chars().iterator();
        i = 0;
        while (chars.hasNext()) {
            var n = charToNum(chars.next());
            if (n != -1) {
                output[(config.inputLength() - i - 1) * 27 + n] = 1f;
            }
            i++;
        }

        return output;
    }

    private static int charToNum(int n) {
        // To uppercase
        if (n >= 'A' && n <= 'Z') {
            n += 'a' - 'A';
        }
        if (n >= 'a' && n <= 'z') {
            return n - 'a';
        } else if (n == ' ') {
            return -1;
        } else {
            return 26;
        }
    }
}
