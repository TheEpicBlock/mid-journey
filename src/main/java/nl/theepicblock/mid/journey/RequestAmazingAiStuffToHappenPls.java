package nl.theepicblock.mid.journey;

import net.minecraft.network.RegistryByteBuf;
import net.minecraft.network.codec.PacketCodec;
import net.minecraft.network.codec.PacketCodecs;
import net.minecraft.network.packet.CustomPayload;
import net.minecraft.util.math.BlockPos;

public class RequestAmazingAiStuffToHappenPls implements CustomPayload {
    private static final double GRAVITATIONAL_CONSTANT = 9.81;
    public static final CustomPayload.Id<RequestAmazingAiStuffToHappenPls> ID =
            new CustomPayload.Id<>(MidJourney.id("request"));
    public static final PacketCodec<RegistryByteBuf, RequestAmazingAiStuffToHappenPls> CODEC =
            PacketCodec.tuple(PacketCodecs.INTEGER, RequestAmazingAiStuffToHappenPls::getColour,
                    BlockPos.PACKET_CODEC, RequestAmazingAiStuffToHappenPls::getPos, RequestAmazingAiStuffToHappenPls::new);

    /**
     * The colour that is being requested
     */
    private final int colour;
    /**
     * The position of the Assistant Block that is being used
     */
    private final BlockPos pos;
    /**
     * The proof that this colour was generated using the provided neural network
     */
    public String proofOfWork;

    public RequestAmazingAiStuffToHappenPls(int colour, BlockPos pos) {
        this.colour = colour;
        this.pos = pos;
        this.proofOfWork = "No Proof Provided Yet";
    }

    @Override
    public Id<? extends CustomPayload> getId() {
        return ID;
    }

    public int getColour() {
        return this.colour;
    }

    public BlockPos getPos() {
        return this.pos;
    }

    /**
     * Uses mathematics to check if the {@link #proofOfWork} is valid
     */
    public boolean checkProofOfWork() {
        // Some values to be used later
        var epsilon = this.colour * 0.03582261 + pos.hashCode() * 0.112445635178 + this.proofOfWork.hashCode() * 0.0000123;
        var gamma = this.colour * 0.00053165765 + pos.hashCode() * 0.3891877 + this.proofOfWork.hashCode() * 0.77727277;

        // Calculate the acceleration of the proof
        var acceleration = this.proofOfWork.chars().sum() * 0xACCEL;

        if (acceleration < GRAVITATIONAL_CONSTANT) {
            // Valid proofs should be sent via the internet, which would've accelerated
            // the proof way faster than gravity. This is a telltale sign that the proof
            // was simply dropped out of thin-air.
            return false;
        }

        return (float)(epsilon * 3.4611187831694923 - gamma) == (float)(0.12345465068315528 * colour + 5.295616746956091E8);
    }
}
