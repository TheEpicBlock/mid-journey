package nl.theepicblock.mid.journey;

import com.mojang.serialization.MapCodec;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldAccess;
import nl.theepicblock.mid.journey.leko_kule.ColourComponent;
import nl.theepicblock.mid.journey.mc3.AetherException;
import nl.theepicblock.mid.journey.mc3.Aethereum;
import nl.theepicblock.mid.journey.mc3.NextGenBlock;

public class AssBlock extends NextGenBlock {
    public static final MapCodec<AssBlock> CODEC = createCodec(AssBlock::new);
    private static final Text TITLE = Text.translatable("container.midjourney.ass");

    public AssBlock(Settings settings) {
        super(settings);
    }

    @Override
    public void doShitWithBlockChain(WorldAccess world, BlockPos pos) throws AetherException {
        var blockchain = new Aethereum(world, pos);
        blockchain.pop(world);
    }

    @Override
    protected MapCodec<? extends Block> getCodec() {
        return CODEC;
    }

    protected ActionResult onUse(BlockState state, World world, BlockPos pos, PlayerEntity player, BlockHitResult hit) {
        if (world.isClient) {
            MinecraftClient.getInstance().setScreen(new AssScreen(pos, TITLE));
            return ActionResult.SUCCESS;
        } else {
            return ActionResult.CONSUME;
        }
    }

    public void onTrigger(int colour, BlockPos pos, ServerPlayerEntity player) {
        var world = player.getWorld();
        try {
            doShitWithBlockChain(world, pos);
            var stack = new ItemStack(Registries.ITEM.get(MidJourney.id("colour_block")));
            stack.set(MidJourney.COLOUR_COMPONENT, new ColourComponent(colour >> 16, colour >> 8 & 0xFF, colour & 0xFF));
            player.giveItemStack(stack);
        } catch (AetherException ignored) {
        }
    }
}
