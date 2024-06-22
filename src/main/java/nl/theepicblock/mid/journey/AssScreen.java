package nl.theepicblock.mid.journey;

import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayNetworking;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import net.minecraft.util.math.BlockPos;
import nl.theepicblock.mid.journey.nn.NeuralNetwork;

@Environment(EnvType.CLIENT)
public class AssScreen extends Screen {
    private static final Identifier TEXTURE = MidJourney.id("textures/gui/ass.png");
    private final BlockPos assistantPos;
    private TextFieldWidget input;
    private ButtonWidget activationButton;
    private int currentColour = 0;
    protected int backgroundWidth = 176;
    protected int backgroundHeight = 78;
    protected int x;
    protected int y;

    public AssScreen(BlockPos assistantPos, Text title) {
        super(title);
        this.assistantPos = assistantPos;
    }

    @Override
    protected void init() {
        super.init();
        this.x = (this.width - this.backgroundWidth) / 2;
        this.y = (this.height - this.backgroundHeight) / 2;

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

        activationButton = ButtonWidget.builder(
                Text.translatable("container.midjourney.ass_activate"),
                (a) -> this.sendPacket()
        ).dimensions(
                this.x + this.backgroundWidth - 5 - 98, this.y + this.backgroundHeight - 5 - 20,
                98, 20
        ).build();
        this.addSelectableChild(this.activationButton);
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

    private void sendPacket() {
        var packet = new RequestAmazingAiStuffToHappenPls(this.currentColour, this.assistantPos);
        // Unhackable
        packet.proofOfWork = "I solemnly swear that this value was created in accordance to the weights and biases provided in the official jar";
        ClientPlayNetworking.send(packet);
    }

    protected void setInitialFocus() {
        this.setInitialFocus(this.input);
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        context.fill(this.x + 5, this.y, this.x + this.backgroundWidth - 5, this.y + this.backgroundHeight, currentColour);
        context.drawTexture(TEXTURE, this.x, this.y, 0, 0, this.backgroundWidth, this.backgroundHeight, this.backgroundWidth, this.backgroundHeight);
        this.input.render(context, mouseX, mouseY, delta);
        this.activationButton.render(context, mouseX, mouseY, delta);
    }

    @Override
    public boolean shouldPause() {
        return false;
    }
}
