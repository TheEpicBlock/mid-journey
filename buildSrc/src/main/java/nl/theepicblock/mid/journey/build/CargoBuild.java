package nl.theepicblock.mid.journey.build;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.Strictness;
import org.gradle.api.DefaultTask;
import org.gradle.api.file.DirectoryProperty;
import org.gradle.api.tasks.*;
import org.gradle.work.DisableCachingByDefault;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Objects;
import java.util.function.Supplier;

@DisableCachingByDefault(because = "Cargo handles detecting if the rust source changed")
public abstract class CargoBuild extends DefaultTask {
    @InputDirectory
    public abstract DirectoryProperty getCrateRoot();

    @OutputDirectory
    public abstract DirectoryProperty getOutput();

    @TaskAction
    public void enact() throws IOException, InterruptedException {
        var process = new ProcessBuilder()
                .directory(getCrateRoot().get().getAsFile())
                .command("cargo", "build", "--quiet", "--message-format=json")
                .redirectOutput(ProcessBuilder.Redirect.PIPE)
                .start();

        try (var output = process.inputReader()) {
            var gson = new GsonBuilder().setStrictness(Strictness.LENIENT).create();
            var outputInfo = output
                    .lines()
                    .map(line -> gson.fromJson(line, JsonObject.class))
                    .filter(line -> Objects.equals(line.get("reason").getAsString(), "compiler-artifact"))
                    .filter(line -> line.has("executable"))
                    .reduce((line, line2) -> line2); // Effectively gives the last line matching the filter

            var executable = Path.of(outputInfo.orElseThrow().get("executable").getAsString());
            var copiedExecutable = getOutput().get().file(executable.getFileName().toString()).getAsFile().toPath();

            // Copy to where gradle wants it (only if needed)
            Supplier<Boolean> needsCopy = () -> {
                try {
                    var attrA = Files.readAttributes(executable, BasicFileAttributes.class);
                    var attrB = Files.readAttributes(copiedExecutable, BasicFileAttributes.class);
                    return !Objects.equals(attrA.creationTime(), attrB.creationTime());
                } catch (IOException e) {
                    return true;
                }
            };

            if (!Files.exists(copiedExecutable) || needsCopy.get()) {
                Files.copy(
                        executable,
                        copiedExecutable,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.COPY_ATTRIBUTES
                );
            }
        }

        if (process.waitFor() != 0) {
            throw new RuntimeException("Failed to build "+getCrateRoot().get());
        }
    }

    public static String getExecutableFromDir(DirectoryProperty dir) {
        var file = dir.getAsFile().get();
        return Objects.requireNonNull(file.listFiles())[0].toString();
    }
}