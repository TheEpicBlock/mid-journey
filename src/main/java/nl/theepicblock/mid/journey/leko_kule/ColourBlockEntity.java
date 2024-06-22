package nl.theepicblock.mid.journey.leko_kule;

import net.fabricmc.fabric.api.blockview.v2.RenderDataBlockEntity;
import net.minecraft.block.BlockState;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.component.ComponentMap;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtOps;
import net.minecraft.network.packet.s2c.play.BlockEntityUpdateS2CPacket;
import net.minecraft.registry.RegistryWrapper;
import net.minecraft.util.math.BlockPos;
import nl.theepicblock.mid.journey.MidJourney;
import org.jetbrains.annotations.Nullable;

public class ColourBlockEntity extends BlockEntity implements RenderDataBlockEntity {
    private ColourComponent colour;

    public ColourBlockEntity(BlockPos pos, BlockState state) {
        super(MidJourney.COLOUR_BLOCK_ENTITY, pos, state);
    }

    @Override
    protected void readComponents(BlockEntity.ComponentsAccess components) {
        super.readComponents(components);
        this.colour = components.getOrDefault(MidJourney.COLOUR_COMPONENT, ColourComponent.DEFAULT);
    }

    @Override
    protected void addComponents(ComponentMap.Builder componentMapBuilder) {
        super.addComponents(componentMapBuilder);
        componentMapBuilder.add(MidJourney.COLOUR_COMPONENT, this.colour);
    }

    @Override
    protected void writeNbt(NbtCompound nbt, RegistryWrapper.WrapperLookup registryLookup) {
        super.writeNbt(nbt, registryLookup);
        if (this.colour != null) {
            nbt.put("colour", ColourComponent.CODEC.encodeStart(registryLookup.getOps(NbtOps.INSTANCE), this.colour).getOrThrow());
        }
    }

    @Override
    protected void readNbt(NbtCompound nbt, RegistryWrapper.WrapperLookup registryLookup) {
        super.readNbt(nbt, registryLookup);
        if (nbt.contains("colour")) {
            ColourComponent.CODEC.parse(registryLookup.getOps(NbtOps.INSTANCE), nbt.get("colour")).resultOrPartial((colour) -> {
                MidJourney.LOGGER.error("Failed to parse colour: '{}'", colour);
            }).ifPresent((colour) -> {
                this.colour = colour;
            });
        }
    }

    @Override
    public BlockEntityUpdateS2CPacket toUpdatePacket() {
        return BlockEntityUpdateS2CPacket.create(this);
    }

    @Override
    public NbtCompound toInitialChunkDataNbt(RegistryWrapper.WrapperLookup registryLookup) {
        return this.createNbt(registryLookup);
    }

    @Override
    public @Nullable Object getRenderData() {
        return this.colour;
    }
}
