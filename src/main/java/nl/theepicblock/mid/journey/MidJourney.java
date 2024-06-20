package nl.theepicblock.mid.journey;

import net.fabricmc.api.ModInitializer;

import net.minecraft.block.AbstractBlock;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.resource.featuretoggle.FeatureFlags;
import net.minecraft.screen.ScreenHandlerType;
import net.minecraft.util.Identifier;
import nl.theepicblock.mid.journey.screen.AssScreenHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MidJourney implements ModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("mid-journey");

	public static final Block ASS_BLOCK = new AssBlock(AbstractBlock.Settings.copy(Blocks.IRON_BLOCK));
	public static final ScreenHandlerType<AssScreenHandler> ASS_SCREEN_HANDLER_SCREEN_HANDLER_TYPE = Registry.register(Registries.SCREEN_HANDLER, id("ass"), new ScreenHandlerType<>((id, inv) -> new AssScreenHandler(id), FeatureFlags.VANILLA_FEATURES));

	@Override
	public void onInitialize() {
		Registry.register(Registries.BLOCK, id("ass_block"), ASS_BLOCK);
	}

	public static Identifier id(String path) {
		return Identifier.of("mid-journey", path);
	}
}