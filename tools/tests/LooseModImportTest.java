package io.github.iamwilmar.mhriseport;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

// Run with app_process in a disposable data directory. Its rutas.txt may point at a read-only game dump.
public final class LooseModImportTest {
    public static void main(String[] args) throws Exception {
        File data = new File(args[0]);
        if (ModFiles.modsDir(data).exists())
            throw new IOException("Use a fresh test directory");
        for (int i = 1; i < args.length; i++) {
            File archive = new File(args[i]);
            boolean reject = archive.getName().startsWith("reject-");
            int before = ModFiles.list(data).size();
            try (FileInputStream in = new FileInputStream(archive)) {
                System.out.println(ModFiles.importArchive(in, archive.getName(), data).description);
                if (reject)
                    throw new AssertionError("Accepted invalid archive");
            } catch (IOException error) {
                if (!reject)
                    throw error;
                if (ModFiles.list(data).size() != before)
                    throw new AssertionError("Partially installed archive");
                System.out.println("Rejected " + archive.getName() + ": " + error.getMessage());
            }
        }
        PcLooseMod.prepareSlots(data);
        List<ModFiles.Mod> mods = ModFiles.list(data);
        long expected = java.util.Arrays.stream(args)
                            .skip(1)
                            .filter(p -> !new File(p).getName().startsWith("reject-"))
                            .count();
        if (mods.size() != expected)
            throw new AssertionError("Lost mod");
        if (mods.size() > 1) {
            ModFiles.Mod first = mods.get(0);
            if (!ModFiles.setOn(first, false))
                throw new AssertionError("Disable failed");
            PcLooseMod.prepareSlots(data);
            if (!ModFiles.setOn(first, true))
                throw new AssertionError("Enable failed");
            PcLooseMod.prepareSlots(data);
        }
    }
}
