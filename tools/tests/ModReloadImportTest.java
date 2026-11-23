package io.github.iamwilmar.mhriseport;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

// Runs on a desktop JVM using android.jar. Exercises the installer after archive extraction.
public final class ModReloadImportTest {
    private static ModFiles.ImportResult install(Path data, String name, String... paths) throws Exception {
        Path work = Files.createTempDirectory(data, ".work-");
        Map<String, File> files = new LinkedHashMap<>();
        for (String path : paths) {
            Path payload = Files.createTempFile(work, "entry", "");
            Files.writeString(payload, path);
            files.put("wrapper/" + path, payload.toFile());
        }
        ModFiles.ImportResult result = ModFiles.installUnpacked(files, name + ".zip", data.toFile(), work.toFile());
        Path dest = ModFiles.modsDir(data.toFile()).toPath().resolve(name);
        for (String path : paths)
            if (!Files.readString(dest.resolve(path)).equals(path)) throw new AssertionError(path);
        return result;
    }

    public static void main(String[] args) throws Exception {
        Path data = Files.createTempDirectory(Path.of(args[0]), "mod-reload-");
        Files.createDirectories(ModFiles.modsDir(data.toFile()).toPath());
        String pak = "romfs/re_chunk_000.pak.patch_016.pak";
        var script = install(data, "script", "reframework/autorun/example.lua");
        var resource = install(data, "resource", pak);
        var mixed = install(data, "mixed", pak, "reframework/data/LuaBoneSystem/model.json");
        if (!script.scripts || script.gameFiles || resource.scripts || !resource.gameFiles ||
                !mixed.scripts || !mixed.gameFiles) throw new AssertionError("Incorrect reload classification");
        System.out.println("Pure Lua, resource-only and mixed imports passed");
    }
}
