package nl.theepicblock.mid.journey;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.networking.v1.PayloadTypeRegistry;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.block.AbstractBlock;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MidJourney implements ModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("mid-journey");

	public static final Block ASS_BLOCK = new AssBlock(AbstractBlock.Settings.copy(Blocks.IRON_BLOCK));

	@Override
	public void onInitialize() {
		Registry.register(Registries.BLOCK, id("ass_block"), ASS_BLOCK);
		PayloadTypeRegistry.playC2S().register(RequestAmazingAiStuffToHappenPls.ID, RequestAmazingAiStuffToHappenPls.CODEC);
		ServerPlayNetworking.registerGlobalReceiver(
				RequestAmazingAiStuffToHappenPls.ID,
				(request, ctx) -> {
					// Reach is about 5 blocks right? Doesn't matter, just means you can't try and access a block
					// at world border. We should also check if the proof of work is correct
					if (ctx.player().getBlockPos().isWithinDistance(request.getPos(), 5.0) && request.checkProofOfWork()) {
						var world = ctx.player().getServerWorld();
						var block = world.getBlockState(request.getPos());
						if (block.getBlock() instanceof AssBlock assistantBlock) {
							assistantBlock.onTrigger(request.getColour(), request.getPos(), ctx.player());
						}
					}
				}
		);
	}

	public static Identifier id(String path) {
		return Identifier.of("mid-journey", path);
	}
}