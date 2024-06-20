package nl.theepicblock.mid.journey.screen;

import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.screen.ScreenHandler;
import nl.theepicblock.mid.journey.MidJourney;

public class AssScreenHandler extends ScreenHandler {
    public AssScreenHandler(int syncId) {
        super(MidJourney.ASS_SCREEN_HANDLER_SCREEN_HANDLER_TYPE, syncId);
    }

    @Override
    public ItemStack quickMove(PlayerEntity player, int slot) {
        return ItemStack.EMPTY;
    }

    @Override
    public boolean canUse(PlayerEntity player) {
        // AI is usable by anyone, that's why AI can solve racism
        return true;
    }
}
