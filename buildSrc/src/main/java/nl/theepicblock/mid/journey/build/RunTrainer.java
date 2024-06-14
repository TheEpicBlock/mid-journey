package nl.theepicblock.mid.journey.build;

import org.gradle.api.DefaultTask;
import org.gradle.api.file.DirectoryProperty;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.tasks.*;

import java.io.File;
import java.io.IOException;

public abstract class RunTrainer extends DefaultTask {
    @InputDirectory
    public abstract DirectoryProperty getBinary();

    @InputFile
    public abstract RegularFileProperty getTrainingData();

    @InputFile
    public abstract RegularFileProperty getConfig();

    @OutputFile
    public abstract RegularFileProperty getOutput();

    @TaskAction
    public void enact() throws IOException, InterruptedException {
        System.out.println(CargoBuild.getExecutableFromDir(getBinary()));
        var process = new ProcessBuilder()
                .command(
                        CargoBuild.getExecutableFromDir(getBinary()),
                        getTrainingData().get().toString(),
                        getConfig().get().toString(),
                        getOutput().get().toString()
                )
                .redirectOutput(new File(getOutput().getAsFile().get() + ".stdout"))
                .redirectError(new File(getOutput().getAsFile().get() + ".stderr"))
                .start();

        if (process.waitFor() != 0) {
            throw new RuntimeException("Failed to run "+getBinary().get());
        }
    }
}