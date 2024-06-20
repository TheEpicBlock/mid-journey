package nl.theepicblock.mid.journey.mc3;

import net.minecraft.block.Block;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldAccess;

/**
 * Just like web3 revolutionized the web, mc3 will revolutionize minecrafting.
 * The {@link NextGenBlock} contains a modern reïmagining of what it means for a block to
 * exist, which is compatible with the core values of mc3.
 * <p>
 * For backwards compatibility, {@link NextGenBlock} extends the legacy mc1 {@link Block} interface. Note that
 * these methods are intended for legacy compatibility only: we at Mid Journey believe that {@link NextGenBlock}
 * contains all the tools necessary to make a blazingly fast block. In the future, support for legacy {@link Block}
 * methods might be removed entirely in favour of the tools contained in {@link NextGenBlock}.
 */
public abstract class NextGenBlock extends Block {
    public NextGenBlock(Settings settings) {
        super(settings);
    }

    public abstract void doShitWithBlockChain(WorldAccess world, BlockPos pos);
}
