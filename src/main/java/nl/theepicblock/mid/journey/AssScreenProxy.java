package nl.theepicblock.mid.journey;

import net.minecraft.client.MinecraftClient;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;

public class AssScreenProxy {
    private static final Text TITLE = Text.translatable("container.midjourney.ass");
    static void openAssScreen(BlockPos pos) {
        MinecraftClient.getInstance().setScreen(new AssScreen(pos, TITLE));
    }
}
