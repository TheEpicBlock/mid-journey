package nl.theepicblock.mid.journey.build;

import com.google.gson.Gson;
import org.gradle.api.DefaultTask;
import org.gradle.api.file.RegularFileProperty;
import org.gradle.api.provider.Property;
import org.gradle.api.tasks.Input;
import org.gradle.api.tasks.OutputFile;
import org.gradle.api.tasks.TaskAction;

import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public abstract class FetchWikipedia extends DefaultTask {
    /**
     * The wikipedia revision id
     */
    @Input
    public abstract Property<String> getRevision();

    @OutputFile
    public abstract RegularFileProperty getOutput();

    @TaskAction
    public void enact() throws IOException, URISyntaxException {
        var wikipediaUrl = new URI("https://en.wikipedia.org/w/rest.php/v1/revision/"+getRevision().get()).toURL();
        var con = (HttpURLConnection)wikipediaUrl.openConnection();
        con.setRequestProperty("Content-Type", "application/json");
        con.setRequestProperty("User-Agent", "MidJourney BuildScript/1.0 (https://github.com/TheEpicBlock/mid-journey; https://theepicblock.nl) "+con.getRequestProperty("User-Agent"));
        con.connect();

        if (con.getResponseCode() != 200) {
            throw new RuntimeException("Statuscode "+con.getResponseCode()+" failed to fetch "+wikipediaUrl);
        }

        var revision = new Gson().fromJson(new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8), WikipediaRevision.class);

        Files.writeString(getOutput().get().getAsFile().toPath(), revision.source());
    }
}