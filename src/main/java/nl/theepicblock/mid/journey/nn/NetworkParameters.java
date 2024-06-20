package nl.theepicblock.mid.journey.nn;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;

import java.io.Reader;

public record NetworkParameters(float[] weights, float[] biases) {
    public static NetworkParameters[] load(Reader stream) {
        var gson = new GsonBuilder().create();
        return gson.fromJson(stream, NetworkParameters[].class);
    }
}
