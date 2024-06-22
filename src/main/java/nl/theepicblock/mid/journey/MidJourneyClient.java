package nl.theepicblock.mid.journey;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.fabricmc.fabric.api.client.rendering.v1.ColorProviderRegistry;
import net.fabricmc.fabric.mixin.client.rendering.BlockColorsMixin;
import net.minecraft.block.BlockState;
import net.minecraft.client.color.block.BlockColorProvider;
import net.minecraft.client.color.item.ItemColorProvider;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.BlockRenderView;
import nl.theepicblock.mid.journey.leko_kule.ColourComponent;
import nl.theepicblock.mid.journey.nn.NNConfig;
import nl.theepicblock.mid.journey.nn.NetworkParameters;
import org.jetbrains.annotations.Nullable;

import java.io.InputStreamReader;

@Environment(EnvType.CLIENT)
public class MidJourneyClient implements ClientModInitializer {
    public static final NNConfig NN_CONFIG;
    public static final NetworkParameters[] NN_PARAMETERS;

    @Override
    public void onInitializeClient() {
        ColorProviderRegistry.BLOCK.register((state, world, pos, tintIndex) -> {
            if (world != null && world.getBlockEntityRenderData(pos) instanceof ColourComponent c) {
                return c.asTint();
            } else {
                return 0;
            }
        }, MidJourney.COLOUR_BLOCK);
        ColorProviderRegistry.ITEM.register((stack, tintIndex) -> stack.getOrDefault(MidJourney.COLOUR_COMPONENT, ColourComponent.DEFAULT).asTint(), Registries.ITEM.get(MidJourney.id("colour_block")));
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
