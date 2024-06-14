package nl.theepicblock.mid.journey.build;

import org.gradle.api.DefaultTask;
import org.gradle.api.file.DirectoryProperty;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.tasks.InputDirectory;
import org.gradle.api.tasks.InputFile;
import org.gradle.api.tasks.OutputDirectory;
import org.gradle.api.tasks.TaskAction;
import org.gradle.work.DisableCachingByDefault;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

@DisableCachingByDefault(because = "If it's cached it won't display the line telling you what to run :(")
public abstract class RunCli extends DefaultTask {
    @InputDirectory
    public abstract DirectoryProperty getBinary();

    @InputFile
    public abstract RegularFileProperty getConfig();

    @InputFile
    public abstract RegularFileProperty getNetwork();

    @OutputDirectory
    public abstract DirectoryProperty getOutput();

    @TaskAction
    public void enact() throws IOException {
        var str = '"'+CargoBuild.getExecutableFromDir(getBinary())+"\" \""+getNetwork().get()+"\" \""+getConfig().get()+'"';
        var name = getBinary().get().getAsFile().getName();

        File output;
        if (System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win")) {
            output = getOutput().get().file(name+".bat").getAsFile();
        } else {
            output = getOutput().get().file(name+".sh").getAsFile();
        }

        Files.writeString(output.toPath(), str, StandardOpenOption.CREATE);
        output.setExecutable(true);

        System.out.println("Please execute "+Path.of(System.getProperty("user.dir")).relativize(output.toPath()));
    }
}