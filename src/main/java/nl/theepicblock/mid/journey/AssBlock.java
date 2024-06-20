package nl.theepicblock.mid.journey;

import com.mojang.serialization.MapCodec;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.screen.NamedScreenHandlerFactory;
import net.minecraft.screen.SimpleNamedScreenHandlerFactory;
import net.minecraft.stat.Stats;
import net.minecraft.text.Text;
import net.minecraft.util.ActionResult;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldAccess;
import nl.theepicblock.mid.journey.mc3.NextGenBlock;
import nl.theepicblock.mid.journey.screen.AssScreenHandler;

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
            return ActionResult.SUCCESS;
        } else {
            player.openHandledScreen(state.createScreenHandlerFactory(world, pos));
            player.incrementStat(Stats.INTERACT_WITH_CRAFTING_TABLE);
            return ActionResult.CONSUME;
        }
    }

    protected NamedScreenHandlerFactory createScreenHandlerFactory(BlockState state, World world, BlockPos pos) {
        return new SimpleNamedScreenHandlerFactory((syncId, inventory, player) -> new AssScreenHandler(syncId), TITLE);
    }
}
