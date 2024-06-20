package nl.theepicblock.mid.journey;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.minecraft.client.gui.screen.ingame.HandledScreens;
import nl.theepicblock.mid.journey.screen.AssScreen;

import java.io.InputStreamReader;
import java.util.Objects;

@Environment(EnvType.CLIENT)
public class MidJourneyClient implements ClientModInitializer {
    public static final NNConfig NN_CONFIG;

    @Override
    public void onInitializeClient() {
        HandledScreens.register(MidJourney.ASS_SCREEN_HANDLER_SCREEN_HANDLER_TYPE, AssScreen::new);
    }

    static {
        var stream = MidJourneyClient.class.getResourceAsStream("/nn_config.json");
        if (stream == null) {
            throw new RuntimeException("Couldn't find config file in jar resources");
        }
        NN_CONFIG = NNConfig.load(new InputStreamReader(stream));
    }
}
