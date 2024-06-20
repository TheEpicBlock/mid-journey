package nl.theepicblock.mid.journey.screen;

import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.ingame.HandledScreen;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import nl.theepicblock.mid.journey.MidJourney;
import nl.theepicblock.mid.journey.MidJourneyClient;
import nl.theepicblock.mid.journey.nn.NeuralNetwork;

@Environment(EnvType.CLIENT)
public class AssScreen extends HandledScreen<AssScreenHandler> {
    private static final Identifier TEXTURE = MidJourney.id("textures/gui/ass.png");
    private TextFieldWidget input;
    private int currentColour = 0;

    public AssScreen(AssScreenHandler handler, PlayerInventory inventory, Text title) {
        super(handler, inventory, title);
        this.backgroundHeight = 78;
    }

    @Override
    protected void init() {
        super.init();
        input = new TextFieldWidget(
                this.textRenderer,
                this.x + 36, this.y + 26,
                103, 12,
                Text.translatable("container.midjourney.ass_tooltip"));
        input.setFocusUnlocked(false);
        input.setEditableColor(-1);
        input.setUneditableColor(-1);
        input.setDrawsBackground(false);
        input.setMaxLength(MidJourneyClient.NN_CONFIG.inputLength());
        input.setChangedListener(this::onTextChanged);
        input.setText("");
        this.addSelectableChild(this.input);
    }

    public void resize(MinecraftClient client, int width, int height) {
        String savedInput = input.getText();
        this.init(client, width, height);
        input.setText(savedInput);
    }

    private void onTextChanged(String newString) {
        // Called when the user edits the texts
        currentColour = NeuralNetwork.eval(newString);
    }

    public boolean keyPressed(int keyCode, int scanCode, int modifiers) {
        assert this.client != null;
        assert this.client.player != null;
        if (keyCode == 256) {
            this.client.player.closeHandledScreen();
        }

        //noinspection SimplifiableConditionalExpression
        return !this.input.keyPressed(keyCode, scanCode, modifiers) && !this.input.isActive() ? super.keyPressed(keyCode, scanCode, modifiers) : true;
    }

    protected void setInitialFocus() {
        this.setInitialFocus(this.input);
    }

    @Override
    protected void drawBackground(DrawContext context, float delta, int mouseX, int mouseY) {
        context.fill(this.x + 5, this.y, this.x + this.backgroundWidth - 5, this.y + this.backgroundHeight, currentColour);
        context.drawTexture(TEXTURE, this.x, this.y, 0, 0, this.backgroundWidth, this.backgroundHeight, this.backgroundWidth, this.backgroundHeight);
        this.input.render(context, mouseX, mouseY, delta);
    }

    @Override
    protected void drawForeground(DrawContext context, int mouseX, int mouseY) {
    }
}
