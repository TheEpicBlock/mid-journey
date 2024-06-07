package nl.theepicblock.mid.journey.build;

import org.gradle.api.DefaultTask;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.tasks.InputFile;
import org.gradle.api.tasks.OutputFile;
import org.gradle.api.tasks.TaskAction;
import org.gradle.work.DisableCachingByDefault;

import java.io.IOException;
import java.nio.file.Files;
import java.util.regex.Pattern;

@DisableCachingByDefault(
        because = "Not worth caching"
)
public abstract class Uncomment extends DefaultTask {
    @InputFile
    public abstract RegularFileProperty getInput();

    @OutputFile
    public abstract RegularFileProperty getOutput();

    private static final Pattern REGEX = Pattern.compile("//.*");

    @TaskAction
    public void enact() throws IOException {
        var input = Files.readString(getInput().get().getAsFile().toPath());
        var outputStr = REGEX.matcher(input).replaceAll("");
        Files.writeString(getOutput().get().getAsFile().toPath(), outputStr);
    }
}
