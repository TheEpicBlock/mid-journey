package nl.theepicblock.mid.journey.leko_kule;

import net.minecraft.block.BlockEntityProvider;
import net.minecraft.block.BlockState;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldAccess;
import nl.theepicblock.mid.journey.mc3.NextGenBlock;
import org.jetbrains.annotations.Nullable;

public class ColourBlock extends NextGenBlock implements BlockEntityProvider {
    public ColourBlock(Settings settings) {
        super(settings);
    }

    @Override
    public void doShitWithBlockChain(WorldAccess world, BlockPos pos) {

    }

    @Nullable
    @Override
    public BlockEntity createBlockEntity(BlockPos pos, BlockState state) {
        return new ColourBlockEntity(pos, state);
    }
}
