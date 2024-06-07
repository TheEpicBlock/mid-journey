package nl.theepicblock.mid.journey.build;

import com.google.gson.Gson;
import org.gradle.api.DefaultTask;
import org.gradle.api.file.RegularFile;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.provider.ListProperty;
import org.gradle.api.tasks.InputFiles;
import org.gradle.api.tasks.OutputFile;
import org.gradle.api.tasks.TaskAction;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.regex.Pattern;

public abstract class ParseWikipedia extends DefaultTask {
    @InputFiles
    public abstract ListProperty<RegularFile> getInput();

    @OutputFile
    public abstract RegularFileProperty getOutput();

    private static final Pattern REGEX = Pattern.compile("\\{\\{Colort/Color.*\\|hex=([^|]+).*\\|name=\\[\\[([^]]+).*}}");

    @TaskAction
    public void enact() throws IOException {
        var output = new HashMap<String, String>();

        for (var file : getInput().get()) {
            var fileContent = Files.readString(file.getAsFile().toPath());
            var matcher = REGEX.matcher(fileContent);

            while (matcher.find()) {
                var colourName = matcher.group(2);
                if (colourName.contains("|")) {
                    colourName = colourName.split("\\|", 2)[1];
                }
                output.put(colourName, matcher.group(1));
            }
        }

        var outputJson = new Gson().toJson(output);
        Files.writeString(getOutput().get().getAsFile().toPath(), outputJson);
    }
}
