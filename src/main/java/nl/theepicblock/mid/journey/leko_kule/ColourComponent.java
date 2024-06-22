package nl.theepicblock.mid.journey.leko_kule;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import io.netty.buffer.ByteBuf;
import net.minecraft.component.type.DyedColorComponent;
import net.minecraft.network.codec.PacketCodec;
import net.minecraft.network.codec.PacketCodecs;

public record ColourComponent(int r, int g, int b) {
    /**
     * This is the colour of "Oh shit something's missing"
     */
    public static final ColourComponent DEFAULT = new ColourComponent(159, 86, 175);
    public static final Codec<ColourComponent> CODEC;
    public static final PacketCodec<ByteBuf, ColourComponent> PACKET_CODEC;

    public int asTint() {
        return r << 16 | g << 8 | b;
    }

    static {
        CODEC = RecordCodecBuilder.create((instance) ->
                instance.group(
                        Codec.INT.fieldOf("r").forGetter(ColourComponent::r),
                        Codec.INT.fieldOf("g").forGetter(ColourComponent::g),
                        Codec.INT.fieldOf("b").forGetter(ColourComponent::b)
                ).apply(instance, ColourComponent::new));
        PACKET_CODEC = PacketCodec.tuple(
                PacketCodecs.VAR_INT, ColourComponent::r,
                PacketCodecs.VAR_INT, ColourComponent::g,
                PacketCodecs.VAR_INT, ColourComponent::b,
                ColourComponent::new);
    }
}
