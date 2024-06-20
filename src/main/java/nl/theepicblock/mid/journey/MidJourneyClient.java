package nl.theepicblock.mid.journey;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import nl.theepicblock.mid.journey.nn.NNConfig;
import nl.theepicblock.mid.journey.nn.NetworkParameters;

import java.io.InputStreamReader;

@Environment(EnvType.CLIENT)
public class MidJourneyClient implements ClientModInitializer {
    public static final NNConfig NN_CONFIG;
    public static final NetworkParameters[] NN_PARAMETERS;

    @Override
    public void onInitializeClient() {
    }

    static {
        var configStream = MidJourneyClient.class.getResourceAsStream("/nn_config.json");
        if (configStream == null) {
            throw new RuntimeException("Couldn't find config file in jar resources");
        }
        NN_CONFIG = NNConfig.load(new InputStreamReader(configStream));
        var parameterStream = MidJourneyClient.class.getResourceAsStream("/network_parameters.json");
        if (parameterStream == null) {
            throw new RuntimeException("Couldn't find parameter file in jar resources");
        }
        NN_PARAMETERS = NetworkParameters.load(new InputStreamReader(parameterStream));
    }
}
