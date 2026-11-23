package io.github.iamwilmar.mhriseport;

import android.content.ContentResolver;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

// The mods folder, as Eden lays out a game's add-ons: <data>/mods/<name>/romfs/... (files that replace the game's,
// fs.cpp) and <data>/mods/<name>/cheats/<build id>.txt (Atmosphere cheats, cheats.cpp). A folder whose name starts with
// '#' is turned off. What the settings menu's Mods section reads and changes.
final class ModFiles {
    private ModFiles() {}

    static final class Mod {
        String name;      // without the '#'
        File dir;
        boolean on;
        boolean romfs, cheats;
    }

    static final class Cheat {
        String name;
        boolean master;
        String key;  // its switch in mhr.ini (cheat_<hash>), the same one cheats.cpp reads
    }

    static File modsDir(File data) {
        return new File(data, "mods");
    }

    static List<Mod> list(File data) {
        List<Mod> out = new ArrayList<>();
        File[] dirs = modsDir(data).listFiles(File::isDirectory);
        if (dirs == null) return out;
        Arrays.sort(dirs, (a, b) -> bare(a.getName()).compareToIgnoreCase(bare(b.getName())));
        for (File d : dirs) {
            if (d.getName().startsWith(".")) continue;
            Mod m = new Mod();
            m.dir = d;
            m.on = !d.getName().startsWith("#");
            m.name = bare(d.getName());
            m.romfs = new File(d, "romfs").isDirectory();
            m.cheats = new File(d, "cheats").isDirectory();
            out.add(m);
        }
        return out;
    }

    private static String bare(String name) {
        return name.startsWith("#") ? name.substring(1) : name;
    }

    // On or off: the folder gets or loses its '#'
    static synchronized boolean setOn(Mod m, boolean on) {
        File to = new File(m.dir.getParentFile(), (on ? "" : "#") + m.name);
        if (to.equals(m.dir)) return true;
        if (to.exists() || !m.dir.renameTo(to)) return false;
        m.dir = to;
        m.on = on;
        return true;
    }

    // What a change to this mod needs: game files only load when the game starts; scripts reload in place
    static boolean gameFiles(File dir) {
        return new File(dir, "romfs").isDirectory() || new File(dir, "exefs").isDirectory();
    }

    static boolean scripts(File dir) {
        return new File(dir, "reframework").isDirectory();
    }

    static synchronized boolean delete(Mod m) {
        return deleteTree(m.dir);
    }

    private static boolean deleteTree(File f) {
        File[] kids = f.listFiles();
        if (kids != null)
            for (File k : kids) deleteTree(k);
        return f.delete();
    }

    // main's build ID (the first 8 bytes of the module ID in its NSO header, +0x40), the name cheat files go by.
    // null if the game's exefs isn't where rutas.txt says.
    static String buildId(File data) {
        DumpScanner.Result r = DumpScanner.load(data);
        File main = new File(r != null && r.exefs != null ? r.exefs : new File(data, "exefs").getPath(), "main");
        try (RandomAccessFile f = new RandomAccessFile(main, "r")) {
            byte[] id = new byte[8];
            f.seek(0x40);
            f.readFully(id);
            StringBuilder b = new StringBuilder();
            for (byte x : id) b.append(String.format(Locale.ROOT, "%02X", x & 0xFF));
            return b.toString();
        } catch (IOException e) {
            return null;
        }
    }

    // The cheat files of a mod for this build (the others are for another game version and don't run)
    static List<File> cheatFiles(Mod m, String buildId) {
        List<File> out = new ArrayList<>();
        File[] files = new File(m.dir, "cheats").listFiles();
        if (files == null || buildId == null) return out;
        Arrays.sort(files);
        for (File f : files) {
            String n = f.getName();
            if (n.toLowerCase(Locale.ROOT).endsWith(".txt") && n.length() >= 20 && n.substring(0, 16).equalsIgnoreCase(buildId))
                out.add(f);
        }
        return out;
    }

    // The mod's cheat files that are for another build
    static int otherBuilds(Mod m, String buildId) {
        File[] files = new File(m.dir, "cheats").listFiles((d, n) -> n.toLowerCase(Locale.ROOT).endsWith(".txt"));
        return files == null ? 0 : files.length - cheatFiles(m, buildId).size();
    }

    static List<Cheat> cheats(Mod m, File file) {
        List<Cheat> out = new ArrayList<>();
        try {
            for (String line : Files.readAllLines(file.toPath(), StandardCharsets.UTF_8)) {
                String l = line.trim();
                if (l.isEmpty() || (l.charAt(0) != '[' && l.charAt(0) != '{')) continue;
                int end = l.indexOf(l.charAt(0) == '[' ? ']' : '}');
                Cheat c = new Cheat();
                c.name = (end < 0 ? l.substring(1) : l.substring(1, end)).trim();
                c.master = l.charAt(0) == '{';
                c.key = key(m.name, c.name);
                out.add(c);
            }
        } catch (IOException ignored) {
            // (unreadable: no cheats)
        }
        return out;
    }

    // FNV-1a of "mod/name" in UTF-8, as cheats.cpp does
    static String key(String mod, String cheat) {
        int h = 0x811c9dc5;
        for (byte b : (mod + "/" + cheat).getBytes(StandardCharsets.UTF_8)) {
            h ^= b & 0xFF;
            h *= 0x01000193;
        }
        return String.format(Locale.ROOT, "cheat_%08x", h);
    }

    // ---- importing ------------------------------------------------------------------------------------------------------

    static String displayName(ContentResolver cr, Uri uri) {
        try (Cursor c = cr.query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (RuntimeException ignored) {
            // (no name: the caller makes one up)
        }
        return null;
    }

    private static String folderName(String file, String fallback) {
        String n = file != null ? file : fallback;
        int dot = n.lastIndexOf('.');
        if (dot > 0) n = n.substring(0, dot);
        if (n.toLowerCase(Locale.ROOT).endsWith(".tar")) n = n.substring(0, n.length() - 4);
        n = n.replaceAll("[\\\\/:*?\"<>|]", "_").trim();
        while (n.startsWith("#") || n.startsWith(".")) n = n.substring(1);
        return n.isEmpty() ? fallback : n;
    }

    private static File freeDir(File data, String name) {
        File d = new File(modsDir(data), name);
        for (int i = 2; d.exists() || new File(modsDir(data), "#" + d.getName()).exists(); i++)
            d = new File(modsDir(data), name + " (" + i + ")");
        return d;
    }

    static synchronized void prepareForLaunch(File data) throws IOException {
        PcLooseMod.prepareSlots(data);
    }

    static ImportResult importArchive(ContentResolver cr, Uri uri, File data) throws IOException {
        try (InputStream in = cr.openInputStream(uri)) {
            if (in == null) throw new IOException("Cannot open archive");
            return importArchive(in, displayName(cr, uri), data);
        }
    }

    static synchronized ImportResult importArchive(InputStream source, String filename, File data) throws IOException {
        File mods = modsDir(data);
        Files.createDirectories(mods.toPath());
        File work = Files.createTempDirectory(mods.toPath(), ".import-").toFile();
        try {
            File archive = new File(work, "archive");
            Files.copy(source, archive.toPath());
            return installUnpacked(ModArchive.unpack(archive, work), filename, data, work);
        } finally {
            deleteTree(work);
        }
    }

    static final class ImportResult {
        final String description;
        final boolean gameFiles, scripts;

        ImportResult(String description, boolean gameFiles, boolean scripts) {
            this.description = description;
            this.gameFiles = gameFiles;
            this.scripts = scripts;
        }
    }

    static ImportResult installUnpacked(Map<String, File> files, String filename, File data, File work) throws IOException {
        File dest = freeDir(data, folderName(filename, "Mod"));
        File staged = new File(work, "ready");
        Files.createDirectory(staged.toPath());
        Map<String, File> pc = PcLooseMod.select(files);
        int scripts = stageFramework(files, staged);
        String description;
        if (!pc.isEmpty()) {
            File game = PcLooseMod.romfs(data);
            int slot = PcLooseMod.nextSlot(data, game);
            File romfs = new File(staged, "romfs");
            Files.createDirectory(romfs.toPath());
            int count = PcLooseMod.convert(pc, game, work, new File(romfs, PcLooseMod.name(slot)));
            Files.write(new File(staged, PcLooseMod.MARKER).toPath(),
                    Long.toString(PcLooseMod.nextOrder(data)).getBytes(StandardCharsets.UTF_8));
            description = count + " PC resources converted";
            long inactive =
                    files.keySet().stream().filter(n -> n.matches("(?i)(?:.*/)?natives/stm/.+")).count() - pc.size();
            if (inactive > 0)
                description += ", " + inactive + " inactive or metadata files skipped";
        } else if (scripts == 0
                || files.keySet().stream().anyMatch(
                        n -> n.matches("(?i)(?:.*/)?(?:natives|romfs|cheats|exefs)/.+"))) {
            description = installSwitch(files, staged);
        } else {
            description = "";
        }
        if (scripts > 0) {
            if (pc.isEmpty())
                Files.write(new File(staged, PcLooseMod.LUA_ORDER).toPath(),
                        Long.toString(PcLooseMod.nextOrder(data)).getBytes(StandardCharsets.UTF_8));
            description += (description.isEmpty() ? "" : ", ") + scripts + " Lua/data files";
        }
        Files.move(staged.toPath(), dest.toPath());
        return new ImportResult("\"" + dest.getName() + "\": " + description, gameFiles(dest), scripts(dest));
    }

    private static int stageFramework(Map<String, File> files, File staged) throws IOException {
        boolean explicit =
                files.keySet().stream().anyMatch(p -> p.matches("(?i)(?:.*/)?reframework/(autorun|data)/.+"));
        String wrapper = null;
        if (!explicit) {
            java.util.regex.Pattern autorun = java.util.regex.Pattern.compile("(?i)^((?:.*/)?)autorun/.+");
            for (String path : files.keySet()) {
                java.util.regex.Matcher m = autorun.matcher(path);
                if (!m.matches())
                    continue;
                if (wrapper != null && !wrapper.equalsIgnoreCase(m.group(1)))
                    throw new IOException("Multiple autorun folders. Select one mod variant.");
                wrapper = m.group(1);
            }
        }
        boolean bare = !explicit && wrapper == null;
        if (bare) {
            for (String path : files.keySet()) {
                if (!path.toLowerCase(Locale.ROOT).endsWith(".lua"))
                    continue;
                String prefix = path.substring(0, path.lastIndexOf('/') + 1);
                if (wrapper == null)
                    wrapper = prefix;
                while (!prefix.startsWith(wrapper))
                    wrapper = wrapper.substring(0, wrapper.lastIndexOf('/', wrapper.length() - 2) + 1);
            }
            if (wrapper == null)
                return 0;
        }
        String prefixPattern = explicit ? "(?:(.*)/)?reframework/" : java.util.regex.Pattern.quote(wrapper);
        java.util.regex.Pattern layout =
                java.util.regex.Pattern.compile("(?i)^" + prefixPattern + "(autorun|data)/(.+)$");
        java.util.Set<String> paths = new java.util.HashSet<>();
        String root = null;
        for (Map.Entry<String, File> entry : files.entrySet()) {
            java.util.regex.Matcher m = layout.matcher(entry.getKey());
            String relative;
            if (m.matches()) {
                int group = explicit ? 2 : 1;
                String prefix = explicit && m.group(1) != null ? m.group(1) : "";
                if (root != null && !root.equalsIgnoreCase(prefix))
                    throw new IOException("Multiple REFramework variants in archive");
                root = prefix;
                relative = m.group(group).toLowerCase(Locale.ROOT) + "/" + m.group(group + 1);
            } else if (bare && entry.getKey().startsWith(wrapper)
                    && entry.getKey().toLowerCase(Locale.ROOT).endsWith(".lua")) {
                relative = "autorun/" + entry.getKey().substring(wrapper.length());
            } else
                continue;
            if (!paths.add(relative.toLowerCase(Locale.ROOT)))
                throw new IOException("Duplicate REFramework file: " + relative);
            File dest = new File(new File(staged, "reframework"), relative);
            Files.createDirectories(dest.getParentFile().toPath());
            Files.copy(entry.getValue().toPath(), dest.toPath());
        }
        return paths.size();
    }

    private static String installSwitch(Map<String, File> files, File staged) throws IOException {
        String prefix = null;
        boolean loose = false;
        for (String marker : new String[]{"romfs|exefs|cheats", "natives"}) {
            java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
                    "^((?:.*/)?)(" + marker + ")/.+$", java.util.regex.Pattern.CASE_INSENSITIVE);
            for (String n : files.keySet()) {
                java.util.regex.Matcher m = pattern.matcher(n);
                if (m.matches() && (prefix == null || m.group(1).length() < prefix.length())) prefix = m.group(1);
            }
            if (prefix != null) break;
            loose = true;
        }
        if (prefix == null) throw new IOException("No natives/STM, romfs, exefs, cheats or natives folder");
        int count = 0;
        boolean exefs = false;
        for (Map.Entry<String, File> e : files.entrySet()) {
            if (!e.getKey().startsWith(prefix)) continue;
            String rest = e.getKey().substring(prefix.length());
            int slash = rest.indexOf('/');
            String first = slash > 0 ? rest.substring(0, slash).toLowerCase(Locale.ROOT) : "";
            if (loose && first.equals("reframework")) continue;
            if (!loose && !Arrays.asList("romfs","exefs","cheats").contains(first)) continue;
            String relative = loose ? "romfs/" + rest : first + rest.substring(slash);
            File out = new File(staged, relative);
            Files.createDirectories(out.getParentFile().toPath());
            Files.move(e.getValue().toPath(), out.toPath());
            count++;
            exefs |= first.equals("exefs");
        }
        return count + " file(s)" + (exefs ? " (exefs patches aren't used by this port)" : "");
    }

    // One cheat file (.txt), into a mod of its own name: mods/<name>/cheats/<build id>.txt. A file named after another
    // build ID is for another version of the game, and isn't taken.
    static String importCheats(ContentResolver cr, Uri uri, File data) throws IOException {
        String buildId = buildId(data);
        if (buildId == null) throw new IOException("can't read the game's build ID (is the exefs still there?)");
        String file = displayName(cr, uri);
        String base = file != null ? file.replaceAll("(?i)\\.txt$", "") : "";
        if (base.matches("[0-9A-Fa-f]{16,64}") && !base.substring(0, 16).equalsIgnoreCase(buildId))
            throw new IOException("that file is for another game version (build " + base.substring(0, 16).toUpperCase(Locale.ROOT)
                    + "; this one is " + buildId + ")");
        byte[] text;
        try (InputStream in = cr.openInputStream(uri)) {
            if (in == null) throw new IOException("can't open it");
            java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int k; (k = in.read(buf)) > 0; ) b.write(buf, 0, k);
            text = b.toByteArray();
        }
        String s = new String(text, StandardCharsets.UTF_8);
        if (!s.contains("[") && !s.contains("{")) throw new IOException("no [cheat] in it");
        File dir = freeDir(data, folderName(base.matches("[0-9A-Fa-f]{16,64}") ? null : file, "Cheats"));
        File cheats = new File(dir, "cheats");
        if (!cheats.mkdirs()) throw new IOException("can't write " + cheats);
        Files.write(new File(cheats, buildId + ".txt").toPath(), text);
        return "\"" + dir.getName() + "\"";
    }
}
