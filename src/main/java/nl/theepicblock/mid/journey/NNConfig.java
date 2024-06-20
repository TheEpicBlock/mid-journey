package nl.theepicblock.mid.journey;


import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;

import java.io.Reader;

public record NNConfig(int inputLength) {
    public static NNConfig load(Reader stream) {
        var gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
        return gson.fromJson(stream, NNConfig.class);
    }
}
