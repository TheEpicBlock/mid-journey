package nl.theepicblock.mid.journey;

import com.mojang.serialization.MapCodec;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldAccess;
import nl.theepicblock.mid.journey.mc3.NextGenBlock;

public class AssBlock extends NextGenBlock {
    public static final MapCodec<AssBlock> CODEC = createCodec(AssBlock::new);
    private static final Text TITLE = Text.translatable("container.midjourney.ass");

    public AssBlock(Settings settings) {
        super(settings);
    }

    @Override
    public void doShitWithBlockChain(WorldAccess world, BlockPos pos) {

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
        player.sendMessage(Text.literal("Please imagine you got colour " + colour + ". I didn't write the coloured blocks yet"));
    }
}
