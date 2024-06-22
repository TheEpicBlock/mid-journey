package nl.theepicblock.mid.journey.mc3;

import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.world.WorldAccess;

import java.util.HashSet;

/**
 * Ethereum was already taken okay.
 */
public class Aethereum {
    private AethereumNode lastNode;

    public Aethereum(WorldAccess world, BlockPos pos) {
        var visited = new HashSet<BlockPos>();
        loop:
        while (true) {
            for (var direction : Direction.values()) {
                if (this.lastNode != null && this.lastNode.prevDir == direction) {
                    continue;
                }
                if (world.getBlockState(pos.offset(direction, 1)).getBlock() != Blocks.CHAIN) {
                    continue;
                }
                if (visited.contains(pos.offset(direction, 2))) {
                    continue;
                }
                if (world.getBlockState(pos.offset(direction, 2)).isSolidBlock(world, pos)) {
                    pos = pos.offset(direction, 2);
                    visited.add(pos);
                    this.lastNode = new AethereumNode(direction.getOpposite(), world.getBlockState(pos), pos, this.lastNode);
                    continue loop;
                }
            }
            break;
        }
    }

    public void pop(WorldAccess world) throws AetherException {
        if (this.lastNode == null) {
            throw new AetherException();
        }
        world.removeBlock(this.lastNode.loc, this.lastNode.prev == null);
        world.breakBlock(this.lastNode.loc.offset(this.lastNode.prevDir, 1), true);
        var node = this.lastNode;
        while (node.prev != null) {
            world.setBlockState(node.prev.loc, node.state, 3);
            node = node.prev;
        }
    }

    public record AethereumNode(Direction prevDir, BlockState state, BlockPos loc, AethereumNode prev) {

    }
}
