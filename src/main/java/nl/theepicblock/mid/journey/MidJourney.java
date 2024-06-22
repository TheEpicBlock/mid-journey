package nl.theepicblock.mid.journey;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.networking.v1.PayloadTypeRegistry;
import net.fabricmc.fabric.api.networking.v1.ServerPlayNetworking;
import net.minecraft.block.AbstractBlock;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.block.entity.BlockEntityType;
import net.minecraft.block.entity.FurnaceBlockEntity;
import net.minecraft.component.ComponentType;
import net.minecraft.component.DataComponentTypes;
import net.minecraft.item.BlockItem;
import net.minecraft.item.Items;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;
import nl.theepicblock.mid.journey.leko_kule.ColourBlock;
import nl.theepicblock.mid.journey.leko_kule.ColourBlockEntity;
import nl.theepicblock.mid.journey.leko_kule.ColourComponent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MidJourney implements ModInitializer {
    public static final Logger LOGGER = LoggerFactory.getLogger("mid-journey");

	public static final Block ASS_BLOCK = new AssBlock(AbstractBlock.Settings.copy(Blocks.IRON_BLOCK));
	public static final Block COLOUR_BLOCK = new ColourBlock(AbstractBlock.Settings.copy(Blocks.WHITE_CONCRETE));
	public static final BlockEntityType<ColourBlockEntity> COLOUR_BLOCK_ENTITY;
	public static final ComponentType<ColourComponent> COLOUR_COMPONENT;

	@Override
	public void onInitialize() {
		Registry.register(Registries.BLOCK, id("ass_block"), ASS_BLOCK);
		Registry.register(Registries.BLOCK, id("colour_block"), COLOUR_BLOCK);
		Items.register(ASS_BLOCK);
		Items.register(COLOUR_BLOCK);

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

	static {
		COLOUR_BLOCK_ENTITY = Registry.register(Registries.BLOCK_ENTITY_TYPE, id("colour_block"), BlockEntityType.Builder.create(ColourBlockEntity::new, COLOUR_BLOCK).build());
		COLOUR_COMPONENT = Registry.register(Registries.DATA_COMPONENT_TYPE, id("colour"), ComponentType.<ColourComponent>builder().codec(ColourComponent.CODEC).packetCodec(ColourComponent.PACKET_CODEC).build());

	}
}