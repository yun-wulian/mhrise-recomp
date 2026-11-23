# Debugging

## The log

`<data>/log.txt`, rewritten on every start. Most lines are in Spanish. Worth knowing:

- **Start-up:** `GPU:` (name, Vulkan version, driver), `custom driver:` (a user driver loaded through
  libadrenotools), `subgrupos: tamaño 64, etapas 0xff, operaciones 0x6ff` (subgroup size, stages, operations), and
  `switch <name>: on/off` the first time each switch is read.
- **`game settings: … is X in the game`** gives the game's own value of each setting the port can touch. A second line
  appears when the port changes it.
- **`N shaders traducidos … M de la caché de SPIR-V`:** translations so far, how many came from the cache, and how many
  failed.
- **`ritmo:`** every few seconds: fps, slow frames, the game's GPU time per frame (`GPU del juego`) and the port's CPU
  time per frame (`ejecutor … ms/frame`). The slowest submit is broken down by cause: pipelines compiled
  (`pipelines del driver`), shaders translated, copies to mirrors (`copia a espejos`), texture uploads.
- **`perfil …`** (only with `perf_stats` or `verbose_log`): where the executor's time goes per frame (textures,
  pipeline, descriptors, state, vertices, draw), compute work, write-backs (`devueltos de overlays`, `entregados`).
- Crashes print the registers, `main+offset` addresses and a backtrace.

`verbose_log` writes line by line (so nothing is lost if it crashes) and adds a watchdog that prints every thread's
stack when the game stops responding (`INSTANTÁNEA`). It's noticeably slower, so turn it off when measuring
performance.

## Frame capture

In the game: Debug → Bug report → **Capture this frame**, then **Save the report**. That makes
`Download/mhrise-report.zip` with `log.txt`, `mhr.ini`, `game_settings.txt`, `rutas.txt` and the captured frame. By
hand, it's a file `<data>/trazar` containing `completa`: the runtime deletes it and captures the next frame.

**Capture button** (`capture_button`, same menu) puts a small draggable button over the game, in a window of its own
the size of the button (`CaptureButton.java`), so the rest of the screen still goes to the touch controls. Tap captures
and waits for `trazar` to disappear (green when done, grey if the game hasn't taken it within 15 seconds); hold saves the
report. Both paths share `BugReport.java`.

The frame goes to `/storage/emulated/0/NieR-volcados/`:

| File | What |
|---|---|
| `fNNNN_ventana.ppm` | The final window image. |
| `fNNNN_rtK.ppm`, `_a.pgm` | Render target K: colour, and alpha as greyscale. |
| `fNNNN_rtK_p16.pgm` | A depth target as 16-bit greyscale. |
| `fNNNN_rtK_cL…` | Layer L of an array target, such as the shadow cascades. |
| `fNNNN_rtK.raw` | Raw bytes, for formats with no picture. |

In the log, that frame's commands start with `F `:

- **`F draw prim … | rt[n] … | vp … | sc …`** for each draw: its targets, viewport and scissor.
- **`F   masks … blend … | depth t1 w0 f7 … | cull … | ms 1 a2c 0 atest 8`:** state. `f` is the compare op: 7 GEQUAL,
  3 EQUAL. `ms` is the sample count, `a2c` alpha-to-coverage, `atest` the alpha test.
- The **vertex streams** and attributes, and a few values from each **uniform buffer**.
- **`vertex <print> pide: …`** and **`pixel <print> …`:** the shaders by fingerprint (the same IDs as in
  `spirv.bin`, `shaders/` and `pipelines_fallidas/`). After that, each **texture** with its handle and **sampler**;
  `sampler 0` where the game expects a real one is a red flag. Then `bindless` lines show how arrays were resolved
  (`(& mask | or)`).
- **`F SSBO leído / escrito`, `F CopyBuffer`, `F dispatch`, `F indirecto`:** buffer traffic and compute work.

To see which passes a frame has, group the `F draw` lines by their `rt[...]` part. To compare two phones, compare the
shaders and samplers in the same pass. That's how the A8xx shadow bug was found: `sampler 0` there, `sampler 268` on
the A725.

To read a PGM or PPM, any image viewer works, or `PIL.Image.open()` in Python. Depth is reversed Z, so near is bright.

**NPC face NaNs (pixel shader `c87ca93efba78188`).** A full frame capture instruments matching HDR draws for that
frame only. Each `face-f<frame>-d<draw>.raw` is the scene colour immediately after that draw; the matching `.spv`
is the original fragment module. A fragment with a nonfinite RGB export encodes its first observed nonfinite input,
sample, division or selected math result. Result 0 means none of those checkpoints saw it. A checkpoint can belong
to a calculation whose value was later discarded. Inspect a known affected face pixel before interpreting the frame
histogram (other draws also occupy this target). Run `python tools/read_face_probe.py <raw> --pixel X Y` in
render-target coordinates, then `spirv-dis --raw-id <spv>` to look up the result ID. The log records matched draws
and successful readbacks. The captured image can show bright diagnostic pixels for that frame.

For the normal-packing compute shader `525985a9ccdd7b21`, a full capture also saves the guest and Vulkan
views of the vertex remap as `normal-remap-f<frame>-cs<dispatch>.{guest,gpu}.bin`. Its S1 input has one
little-endian 32-bit word per vertex. The low 22 bits select a 12-byte entry in S0. The `normal remap` log
line gives the guest address, binding source and buffer sizes for that dispatch.

## GPU breadcrumbs

`breadcrumbs` (Debug menu). Every pass and dispatch writes a marker, and the GPU runs everything in order. When the GPU
hangs, or the driver kills a stuck job, the log says `breadcrumbs: GPU perdida` and names the last pass or dispatch
started, with its shaders. Their code is saved as `<data>/shaders/migas_<print>.code` and `_ctrl.code`, to read with
`tools/shadertool`. `breadcrumbs_compute` and `breadcrumbs_light` are lighter variants. It's slow; turn it off
afterwards.

The control section says how a compute shader runs: workgroup size at `+0x748` (3 × u32), shared memory at `+0x754`.
That's how the Mali hang was traced to a 32-thread shader.

## Probes in the game's code

`<data>/sondas`: one address per line (`main+0x6c50d60`, optionally followed by how many hits to log, 40 by default).
When the game loads, each instruction there is replaced by a jump to a stub. The stub logs x0..x7, x19..x24 and lr with
the thread's name, runs the original instruction and returns. It's for seeing what the game passes to its own
functions. Instructions that depend on the pc can't be probed.

## Shader tools

- `<data>/shaders/`: stages that failed to translate, `migas_*` from breadcrumbs, `barrera_*` (compute with barriers in
  conditional flow).
- `<data>/pipelines_fallidas/`: the SPIR-V of each stage of a pipeline the driver refused (64 per session at most).
- `tools/shadertool`: translates a dumped stage with the same recompiler, offline.
- `tools/revisar_spirv.py`: reads a phone's `spirv.bin`, its `pipelines_fallidas/` or loose `.spv` files, and says
  what each shader uses that tends to break on a given GPU (`--gpu mali-g615|adreno-730|adreno-650`): subgroup ops on
  small subgroups, stores from vertex shaders, shared memory over the limit, NonUniform, 64-bit and half types.
- Useful SPIR-V opcodes: `OpGroupNonUniformBallot` 339, `Shuffle` 345, `ShuffleXor` 346, `ShuffleUp` 347,
  `ShuffleDown` 348, `QuadBroadcast` 365, `QuadSwap` 366. `BuiltIn` 41 is `SubgroupLocalInvocationId`.

## Looking at the game's code

The ExeFS `main` is an NSO with LZ4 segments. `tools/nso.py <exefs> <out>` reads the modules, so a plain image can be disassembled at
`main+offset`; with LLVM: `llvm-objcopy -I binary` to an ELF, then `llvm-objdump -d`. To find an engine property's
setter, find the `set_Name` string and the `adrp`/`add` that loads it in the registration code. The setter's thunk is
loaded into `x2` just before, from a slot the relocations fill (`R_AARCH64_RELATIVE`, info 0x403). That's how every
address in [mh-rise.md](mh-rise.md) was found.

## Other switches for debugging

See the [settings reference](settings.md). The ones used most: `no_fsr` (rules out the upscaler), `no_split_overlays`,
`no_readback_copies` and `use_return` (memory coherence), `no_safe_translation`, `keep_dead_functions`,
`no_spirv_opt_brx` (translation), `no_ubwc` and `env.TU_DEBUG` (driver), and `perf_stats` (CPU time).
