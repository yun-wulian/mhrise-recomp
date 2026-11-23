// HUD de rendimiento (ver hud.h). Todo en el hilo que presenta: sin locks.

#include "nnsdk/nvn/hud.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "nnsdk/host.h"
#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/runtime.h"
#include "nnsdk/ini.h"

namespace nvn::hud {
struct Buffer {
    VkDevice device;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* data = nullptr;
    VkDeviceSize size;
    ~Buffer() {
        if (data) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};
namespace {

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

constexpr uint32_t kW = 44, kH = 26;       // en píxeles del swapchain (1920x1080): solo el número, que no estorbe
constexpr int32_t kX = 8, kY = 8;          // esquina superior izquierda
// The overlay with its other lines: a box at most this big (scale-2 text, 12 x 20 pixels a character and a line)
constexpr uint32_t kStatsW = 360, kStatsH = 170;
constexpr int kStatsLine = 20, kStatsPad = 8, kStatsLines = 8;
std::atomic<int> g_building{0};  // pipelines in the background threads (shader_queued / shader_done)
// El panel de ajustes (ver hud.h): un cajón que sale del borde derecho, de un tercio del ancho de la pantalla y toda su
// altura, con una fila por ajuste.
constexpr int64_t kPanelSlideNs = 180'000'000;    // lo que tarda en salir o esconderse
constexpr int64_t kPanelIdleNs = 8'000'000'000;   // sin tocarlo, se esconde solo
constexpr int kPanelHeader = 150, kPanelRow = 132;  // alto de la cabecera y de cada fila, en píxeles

// Fuente de 5x7: cada fila, 5 bits (el alto a la izquierda).
struct Glyph {
    char c;
    uint8_t rows[7];
};
constexpr Glyph kFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}}, {'%', {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}}, {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}}, {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}}, {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}}, {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}}, {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}}, {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}}, {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}}, {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}}, {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}}, {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}}, {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}}, {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}}, {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
};

struct Rgba {
    uint8_t r, g, b, a;
};
constexpr Rgba kBack{12, 12, 14, 255};
constexpr Rgba kGood{80, 210, 110, 255}, kWarn{235, 195, 60, 255}, kBad{235, 80, 70, 255};
constexpr Rgba kEdge{58, 58, 66, 255}, kText{232, 232, 238, 255}, kOff{74, 74, 84, 255}, kKnob{240, 240, 244, 255};
constexpr Rgba kDim{150, 150, 160, 255}, kLine{34, 34, 40, 255};

// Los textos del panel, en español si el sistema está en español y en inglés si no.
bool spanish() {
#ifdef __ANDROID__
    char value[PROP_VALUE_MAX] = "";
    if (__system_property_get("persist.sys.locale", value) <= 0) __system_property_get("ro.product.locale", value);
    return value[0] == 'e' && value[1] == 's';
#else
    return true;
#endif
}

// El último toque de la pantalla, como lo deja el loader en NnsdkInput (host.h).
struct Touch {
    uint32_t serial = 0, state = 0;  // state: 1 posado, 0 levantado, 2 cancelado
    int sx = 0, sy = 0, x = 0, y = 0;  // dónde se posó y por dónde va (o dónde se levantó), en píxeles de la ventana
};
bool read_touch(Touch* t) {
    NnsdkInput* in = nnsdk::host().input;
    if (!in) return false;
    t->serial = in->touch_serial.load(std::memory_order_acquire);
    t->state = in->touch_state.load(std::memory_order_acquire);
    const uint64_t start = in->touch_start.load(std::memory_order_acquire), last = in->touch_last.load(std::memory_order_acquire);
    t->sx = static_cast<int32_t>(start & 0xffffffffu), t->sy = static_cast<int32_t>(start >> 32);
    t->x = static_cast<int32_t>(last & 0xffffffffu), t->y = static_cast<int32_t>(last >> 32);
    return t->serial == in->touch_serial.load(std::memory_order_acquire);  // si cambió a medias, al frame siguiente
}

class Hud {
public:
    void real_frame(int64_t now) {
        if (last_real_ && now > last_real_) {
            intervals_.push_back((now - last_real_) / 1e6);
            if (intervals_.size() > 300) intervals_.pop_front();
        }
        last_real_ = now;
        reals_.push_back(now);
        trim(reals_, now);
        trim(generated_, now);  // (with frame generation off none are added: the old ones go here)
    }
    void generated_frame() {
        generated_.push_back(last_real_);
        trim(generated_, last_real_);
    }
    void record(Uploads& uploads, VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
        uploads.panel.reset();
        // The counter goes off with no_fps in mhr.ini (live, as always) or from the panel, which writes that same line:
        // the choice also holds the next time the game opens.
        if (const bool file = NVN_FLAG("no_fps"); !flag_seen_ || file != flag_) flag_seen_ = true, flag_ = hidden_ = file;
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        gestures(extent, now);
        if (failed_) return;
        const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
        if (!bgra && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_R8G8B8A8_SRGB) return;
        const bool stats = !NVN_FLAG("no_hud_frametime") || !NVN_FLAG("no_hud_cpu") || !NVN_FLAG("no_hud_gpu") || !NVN_FLAG("no_hud_ram") ||
                           !NVN_FLAG("no_hud_heat") || !NVN_FLAG("no_hud_shaders");
        if (!hidden_ && stats) {
            if (extent.width >= kX + kStatsW && extent.height >= kY + kStatsH &&
                reserve(uploads.counter, kStatsW, kStatsH)) {
                sample(now);
                Buffer& b = *uploads.counter;
                uint32_t bw = 0, bh = 0;
                draw_stats(reinterpret_cast<Rgba*>(b.data), bgra, &bw, &bh);
                VkBufferImageCopy region{};
                region.bufferRowLength = kStatsW;
                region.bufferImageHeight = kStatsH;
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageOffset = {kX, kY, 0};
                region.imageExtent = {bw, bh, 1};
                overlay_copy(cmd, b, image, region);
            }
        } else if (!hidden_ && extent.width >= kX + kW && extent.height >= kY + kH && reserve(uploads.counter, kW, kH)) {
            Buffer& b = *uploads.counter;
            draw(reinterpret_cast<Rgba*>(b.data), bgra);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset = {kX, kY, 0};
            region.imageExtent = {kW, kH, 1};
            overlay_copy(cmd, b, image, region);
        }
        record_panel(uploads, cmd, image, extent, bgra, now);
    }

private:
    static void overlay_copy(VkCommandBuffer cmd, const Buffer& b, VkImage image, const VkBufferImageCopy& region) {
        order_present_writes(cmd, image);
        vkCmdCopyBufferToImage(cmd, b.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    static void trim(std::deque<int64_t>& q, int64_t now) {
        while (!q.empty() && now - q.front() > 1000000000) q.pop_front();
    }
    // El panel: cuánto asoma (0 a 1) en este instante, según cuándo se abrió o se cerró.
    float panel_shown(int64_t now) const {
        const float t = std::clamp(static_cast<float>(now - panel_at_) / kPanelSlideNs, 0.0f, 1.0f);
        const float eased = 1.0f - (1.0f - t) * (1.0f - t);
        return panel_open_ ? panel_from_ + (1.0f - panel_from_) * eased : panel_from_ * (1.0f - eased);
    }
    void set_panel(bool open, int64_t now) {
        if (open == panel_open_) return;
        panel_from_ = panel_shown(now);  // desde donde esté: abrirlo a medio esconderse no da un salto
        panel_open_ = open;
        panel_at_ = now;
        touched_at_ = now;
        if (open) {  // por si los ficheros cambiaron desde fuera
            const bool marked = has_setup() && access(data_file("elegir_carpeta").c_str(), F_OK) == 0;
            if (marked != choose_folder_) choose_folder_ = marked, panel_dirty_ = true;
            const bool tv = has_tv() && nnsdk::ini::on("tv_mode");
            if (tv != tv_) tv_ = tv, panel_dirty_ = true;
            const bool fg = nnsdk::ini::on("lsfg");
            if (fg != fg_) fg_ = fg, panel_dirty_ = true;
        }
    }
    // Las filas del panel, en orden: el contador de FPS y la generación de frames siempre; el modo TV si el port lo tiene;
    // cambiar de carpeta si la app tiene pantalla de inicio.
    enum RowId { kRowFps, kRowFg, kRowTv, kRowFolder };
    static int rows(RowId* out) {
        int n = 0;
        out[n++] = kRowFps;
        out[n++] = kRowFg;
        if (has_tv()) out[n++] = kRowTv;
        if (has_setup()) out[n++] = kRowFolder;
        return n;
    }
    // ¿El port tiene modo TV (la consola en la base)? El loader pone NIER_TV_MODE en los que lo traen.
    static bool has_tv() {
        static const bool tv = getenv("NIER_TV_MODE") != nullptr;
        return tv;
    }
    // <datos>/<name>.
    static std::string data_file(const char* name) {
        const char* data = getenv("NIER_DATA");
        return std::string(data ? data : ".") + "/" + name;
    }
    // ¿La app tiene pantalla de inicio (la que pide la carpeta del juego)? El loader pone NIER_SETUP cuando la carpeta sale
    // de ella. Solo entonces tiene sentido la fila "Cambiar carpeta".
    static bool has_setup() {
        static const bool setup = getenv("NIER_SETUP") != nullptr;
        return setup;
    }
    // Los gestos, una vez por frame presentado.
    // Abrir: el gesto de "atrás" del sistema desde el borde derecho (el de la flecha; Android se queda con ese toque y al
    // juego solo le llega la tecla, que el loader convierte en NnsdkInput::hud_gesture), o un dedo que se posa en la mitad
    // derecha de la pantalla y va hacia la izquierda al menos un octavo de su ancho.
    // Con el panel fuera: un toque en una fila cambia su ajuste; un toque fuera del panel, deslizar hacia la derecha, el
    // mismo gesto del borde o un rato sin tocarlo lo esconden.
    void gestures(VkExtent2D extent, int64_t now) {
        const int w = static_cast<int>(extent.width);
        const int panel_w = w / 3;
        if (NnsdkInput* in = nnsdk::host().input) {
            const uint64_t gesture = in->hud_gesture.load(std::memory_order_acquire);
            if (!gesture_seen_) gesture_seen_ = true, gesture_ = gesture;  // lo de antes de empezar no cuenta
            if (gesture != gesture_) {
                gesture_ = gesture;
                // Si ese mismo toque ya lo abrió deslizando (el sistema tardó en quedárselo), no lo cierra.
                if (!(panel_open_ && opened_by_ == static_cast<uint32_t>(gesture >> 32))) set_panel(!panel_open_, now);
            }
        }
        Touch t;
        if (!read_touch(&t)) return;
        if (!touch_seen_) touch_seen_ = true, touch_handled_ = t.serial;
        if (t.serial != touch_handled_) {
            const int dx = t.x - t.sx, dy = t.y - t.sy;
            const bool ended = t.state != 1;
            if (t.state == 2) {
                touch_handled_ = t.serial;
            } else if (!panel_open_) {
                if (t.sx > w / 2 && dx < -w / 8 && std::abs(dy) * 2 < -dx) {
                    set_panel(true, now);
                    opened_by_ = t.serial;
                    touch_handled_ = t.serial;  // al levantar este mismo dedo no pasa nada más
                } else if (ended) {
                    touch_handled_ = t.serial;
                }
            } else if (dx > w / 12 && std::abs(dy) * 2 < dx) {
                set_panel(false, now);
                touch_handled_ = t.serial;
            } else if (ended) {
                if (std::abs(dx) < 32 && std::abs(dy) < 32) {
                    if (t.x < w - panel_w) {
                        set_panel(false, now);
                    } else {
                        touched_at_ = now;
                        const int row = t.y >= kPanelHeader ? (t.y - kPanelHeader) / kPanelRow : -1;
                        RowId ids[4];
                        if (row >= 0 && row < rows(ids)) {
                            if (ids[row] == kRowFps) set_hidden(!hidden_);
                            else if (ids[row] == kRowFg) set_fg(!fg_);
                            else if (ids[row] == kRowTv) set_tv(!tv_);
                            else set_choose_folder(!choose_folder_);
                        }
                    }
                }
                touch_handled_ = t.serial;
            }
        }
        if (panel_open_ && now - touched_at_ > kPanelIdleNs) set_panel(false, now);
    }
    void set_hidden(bool hidden) {
        hidden_ = hidden;
        panel_dirty_ = true;
        nnsdk::ini::set("no_fps", hidden ? "1" : "0");
        NVN_LOG("hud: contador de FPS %s desde el panel\n", hidden ? "apagado" : "encendido");
    }
    // "Cambiar carpeta": con el fichero <datos>/elegir_carpeta, la pantalla de inicio de la app (que normalmente arranca el
    // juego directamente) vuelve a salir la próxima vez que se abra, para elegir otra carpeta; ella misma borra el fichero.
    void set_choose_folder(bool on) {
        choose_folder_ = on;
        panel_dirty_ = true;
        const std::string file = data_file("elegir_carpeta");
        if (!on) {
            unlink(file.c_str());
        } else if (const int fd = open(file.c_str(), O_WRONLY | O_CREAT, 0644); fd >= 0) {
            close(fd);
        }
        NVN_LOG("hud: cambiar de carpeta al abrir la app: %s\n", on ? "sí" : "no");
    }
    // "Frame generation": the lsfg line of mhr.ini, which the presenter reads live (lsfg/framegen.cpp): on and off at
    // once, and kept for the next start
    void set_fg(bool on) {
        fg_ = on;
        panel_dirty_ = true;
        nnsdk::ini::set("lsfg", on ? "1" : "0");
        NVN_LOG("hud: frame generation %s from the panel\n", on ? "on" : "off");
    }
    // "TV mode": with tv_mode in mhr.ini the game thinks it's docked (more resolution, Pro Controller) from the next
    // time it opens; the mode is decided at start-up (nnsdk::tv_mode).
    void set_tv(bool on) {
        tv_ = on;
        panel_dirty_ = true;
        nnsdk::ini::set("tv_mode", on ? "1" : "0");
        NVN_LOG("hud: modo TV al abrir la app: %s\n", on ? "sí" : "no");
    }
    void record_panel(Uploads& uploads, VkCommandBuffer cmd, VkImage image, VkExtent2D extent, bool bgra, int64_t now) {
        const uint32_t w = extent.width / 3, h = extent.height;
        if (w < 360 || h < static_cast<uint32_t>(kPanelHeader + kPanelRow)) return;
        const uint32_t visible = static_cast<uint32_t>(panel_shown(now) * w + 0.5f);
        if (!visible) return;
        if (panel_w_ != w || panel_h_ != h || panel_bgra_ != bgra) panel_dirty_ = true;
        if (hidden_ != panel_drawn_hidden_) panel_dirty_ = true;  // lo apagaron con el fichero, desde fuera
        if (panel_dirty_) {
            // A submitted copy retains its immutable pixels until its own fence signals.
            if (panel_.use_count() > 1) panel_.reset();
            if (!reserve(panel_, w, h)) return;
            panel_w_ = w;
            panel_h_ = h;
            panel_bgra_ = bgra;
            panel_dirty_ = false;
            panel_drawn_hidden_ = hidden_;
            draw_panel(reinterpret_cast<Rgba*>(panel_->data), bgra);
        }
        // Asoma por el borde derecho: las `visible` primeras columnas del panel, pegadas al borde.
        VkBufferImageCopy region{};
        region.bufferRowLength = w;
        region.bufferImageHeight = h;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {static_cast<int32_t>(extent.width - visible), 0, 0};
        region.imageExtent = {visible, h, 1};
        uploads.panel = panel_;
        overlay_copy(cmd, *panel_, image, region);
    }
    // La cabecera y una fila por ajuste: su nombre y un interruptor (verde con el pomo a la derecha, encendido; gris con el
    // pomo a la izquierda, apagado).
    void draw_panel(Rgba* px, bool bgra) const {
        const int w = static_cast<int>(panel_w_), h = static_cast<int>(panel_h_);
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= w || y >= h) return;
            px[y * w + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int rw, int rh, Rgba c) {
            for (int y = y0; y < y0 + rh; y++)
                for (int x = x0; x < x0 + rw; x++) put(x, y, c);
        };
        const auto pill = [&](int x0, int y0, int pw, int ph, Rgba c) {  // rectángulo con los extremos redondos
            const int r = ph / 2;
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    const int cx = x < r ? r : x >= pw - r ? pw - r - 1 : x, ex = x - cx, ey = y - r;
                    if (ex * ex + ey * ey <= r * r) put(x0 + x, y0 + y, c);
                }
        };
        const auto text = [&](int x, int y, int scale, const char* s, Rgba c) {
            for (; *s; s++, x += 6 * scale) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b)) rect(x + b * scale, y + r * scale, scale, scale, c);
            }
        };
        const bool es = spanish();
        const int margin = 44;
        rect(0, 0, w, h, kBack);
        rect(0, 0, 4, h, kEdge);  // el canto que da al juego
        text(margin, (kPanelHeader - 7 * 6) / 2, 6, es ? "AJUSTES" : "SETTINGS", kText);
        rect(4, kPanelHeader - 3, w - 4, 3, kEdge);
        // Las filas de rows(): el contador de FPS, el modo TV y volver a elegir la carpeta del juego.
        struct Row {
            const char* es;
            const char* en;
            const char* hint_es;  // una línea pequeña debajo (o nullptr)
            const char* hint_en;
            bool on;
        };
        const Row all[] = {{"CONTADOR DE FPS", "FPS COUNTER", nullptr, nullptr, !hidden_},
                           {"GENERAR FRAMES", "FRAME GEN", "X2 A 60 FPS", "X2 TO 60 FPS", fg_},
                           {"MODO TV", "TV MODE", "AL ABRIR LA APP", "AT NEXT START", tv_},
                           {"CAMBIAR CARPETA", "CHANGE FOLDER", "AL ABRIR LA APP", "AT NEXT START", choose_folder_}};
        RowId ids[4];
        const int count = rows(ids);
        int y = kPanelHeader;
        for (int k = 0; k < count; k++) {
            const Row& row = all[ids[k]];
            const int sw = 132, sh = 60, sx = w - sw - margin, sy = y + (kPanelRow - sh) / 2;
            if (row.hint_es) {
                text(margin, y + 30, 4, es ? row.es : row.en, kText);
                text(margin, y + 30 + 7 * 4 + 14, 3, es ? row.hint_es : row.hint_en, kDim);
            } else {
                text(margin, y + (kPanelRow - 7 * 4) / 2, 4, es ? row.es : row.en, kText);
            }
            pill(sx, sy, sw, sh, row.on ? kGood : kOff);
            pill(row.on ? sx + sw - sh + 5 : sx + 5, sy + 5, sh - 10, sh - 10, kKnob);
            rect(margin, y + kPanelRow - 2, w - 2 * margin, 2, kLine);
            y += kPanelRow;
        }
        // Al pie, cómo se cierra.
        text(margin, h - 70, 3, es ? "TOCA FUERA PARA CERRAR" : "TAP OUTSIDE TO CLOSE", kDim);
    }
    bool reserve(std::shared_ptr<Buffer>& buffer, uint32_t width, uint32_t height) {
        const VkDeviceSize size = VkDeviceSize{width} * height * 4;
        if (buffer && buffer->size >= size) return true;
        VkContext* c = vk();
        auto next = std::make_shared<Buffer>();
        next->device = c->device;
        next->size = size;
        Buffer& b = *next;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkMemoryRequirements req;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        if (vkCreateBuffer(c->device, &bi, nullptr, &b.buffer) != VK_SUCCESS) return fail();
        vkGetBufferMemoryRequirements(c->device, b.buffer, &req);
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(c->device, &ai, nullptr, &b.memory) != VK_SUCCESS ||
            vkBindBufferMemory(c->device, b.buffer, b.memory, 0) != VK_SUCCESS ||
            vkMapMemory(c->device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.data) != VK_SUCCESS)
            return fail();
        buffer = std::move(next);
        return true;
    }
    bool fail() {
        NVN_LOG("hud: no se pudo crear su buffer; sin HUD\n");
        failed_ = true;
        return false;
    }

    // A small text file's first bytes (sysfs, /proc), or false if it can't be read from the app
    static bool read_file(const char* path, char* out, size_t size) {
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        const ssize_t n = read(fd, out, size - 1);
        close(fd);
        if (n <= 0) return false;
        out[n] = 0;
        return true;
    }
    // The GPU's busy %, from wherever this phone's driver puts it: Adreno's two kgsl files, or the one many Mali and
    // PowerVR kernels have. -1 where none of them can be read (most phones keep them from apps on newer Android).
    int gpu_busy() {
        static const char* const kPaths[] = {"/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
                                             "/sys/class/kgsl/kgsl-3d0/gpubusy", "/sys/kernel/gpu/gpu_busy"};
        char buf[64];
        for (int k = gpu_source_ >= 0 ? gpu_source_ : 0; k < 3; k++) {
            if (gpu_source_ == -1) break;
            if (!read_file(kPaths[k], buf, sizeof buf)) {
                if (gpu_source_ >= 0) return -1;  // (it was readable a moment ago: a hiccup)
                continue;
            }
            int value = -1;
            if (k == 1) {  // "busy total": cycles in the last sample window
                unsigned long long busy = 0, total = 0;
                if (sscanf(buf, "%llu %llu", &busy, &total) == 2) value = total ? static_cast<int>(busy * 100 / total) : 0;
            } else {
                value = atoi(buf);
            }
            if (value < 0) continue;
            if (gpu_source_ < 0) NVN_LOG("hud: GPU load from %s\n", kPaths[k]);
            gpu_source_ = k;
            return std::min(value, 100);
        }
        if (gpu_source_ == -2) NVN_LOG("hud: this phone doesn't let apps read the GPU load\n");
        gpu_source_ = -1;
        return -1;
    }
    // Everything the lines show, twice a second (a few small reads: nothing next to a frame)
    void sample(int64_t now) {
        if (sampled_at_ && now - sampled_at_ < 500'000'000) return;
        char buf[512];
        uint64_t ticks = 0;
        if (read_file("/proc/self/stat", buf, sizeof buf)) {
            // (after the name in brackets: state, then 10 fields, then utime and stime)
            if (const char* p = strrchr(buf, ')')) {
                unsigned long ut = 0, st = 0;
                if (sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &ut, &st) == 2) ticks = ut + st;
            }
        }
        if (sampled_at_ && ticks >= cpu_ticks_) {
            static const double hz = static_cast<double>(sysconf(_SC_CLK_TCK));
            static const double cores = static_cast<double>(std::max(1L, sysconf(_SC_NPROCESSORS_CONF)));
            const double busy = static_cast<double>(ticks - cpu_ticks_) / hz, wall = (now - sampled_at_) / 1e9;
            cpu_ = static_cast<int>(std::min(100.0, 100.0 * busy / (wall * cores)) + 0.5);
        }
        cpu_ticks_ = ticks;
        sampled_at_ = now;
        if (!NVN_FLAG("no_hud_gpu")) gpu_ = gpu_busy();
        if (read_file("/proc/self/statm", buf, sizeof buf)) {
            unsigned long size = 0, resident = 0;
            if (sscanf(buf, "%lu %lu", &size, &resident) == 2)
                ram_app_ = static_cast<float>(resident * static_cast<double>(getpagesize()) / 1073741824.0);
        }
        if (read_file("/proc/meminfo", buf, sizeof buf)) {
            unsigned long total = 0, available = 0;
            if (const char* t = strstr(buf, "MemTotal:")) sscanf(t + 9, "%lu", &total);
            if (const char* a = strstr(buf, "MemAvailable:")) sscanf(a + 13, "%lu", &available);
            if (total) ram_total_ = total / 1048576.0f, ram_used_ = (total - std::min(total, available)) / 1048576.0f;
        }
        // The game's frames: their average length over the last half second or so, and the 1 % low (the fps of the
        // slowest 1 % of the last 300)
        if (!intervals_.empty()) {
            const size_t n = std::min<size_t>(intervals_.size(), 15);
            double sum = 0;
            for (size_t k = intervals_.size() - n; k < intervals_.size(); k++) sum += intervals_[k];
            frame_ms_ = static_cast<float>(sum / n);
            std::vector<double> sorted(intervals_.begin(), intervals_.end());
            const size_t at = sorted.size() - 1 - sorted.size() / 100;
            std::nth_element(sorted.begin(), sorted.begin() + at, sorted.end());
            low_fps_ = sorted[at] > 0 ? static_cast<float>(1000.0 / sorted[at]) : 0.0f;
        }
    }
    // The box: the FPS (in its colour) and each line that's on. bw, bh: the part of the buffer drawn.
    void draw_stats(Rgba* px, bool bgra, uint32_t* bw, uint32_t* bh) {
        if (last_real_ - text_at_ > 250000000 || !text_at_) {
            text_at_ = last_real_;
            shown_ = static_cast<int>(reals_.size() + generated_.size());
        }
        struct Line {
            char text[40];
            Rgba color;
        };
        Line lines[kStatsLines];
        int n = 0;
        const int target = generated_.empty() ? 30 : 60;
        const Rgba fps_color = shown_ >= target - 1 ? kGood : shown_ >= target * 5 / 6 ? kWarn : kBad;
        if (generated_.empty()) snprintf(lines[n].text, sizeof lines[n].text, "%d FPS", std::min(shown_, 999));
        else snprintf(lines[n].text, sizeof lines[n].text, "%d FPS  %d REAL", std::min(shown_, 999),
                      static_cast<int>(reals_.size()));
        lines[n++].color = fps_color;
        if (!NVN_FLAG("no_hud_frametime")) {
            snprintf(lines[n].text, sizeof lines[n].text, "%.1f MS  1%% LOW %.0f", frame_ms_, low_fps_);
            lines[n++].color = kText;
        }
        if (!NVN_FLAG("no_hud_cpu")) {
            if (cpu_ >= 0) snprintf(lines[n].text, sizeof lines[n].text, "CPU %d%%", cpu_);
            else snprintf(lines[n].text, sizeof lines[n].text, "CPU --");
            lines[n++].color = cpu_ >= 85 ? kWarn : kText;
        }
        if (!NVN_FLAG("no_hud_gpu")) {
            if (gpu_ >= 0) snprintf(lines[n].text, sizeof lines[n].text, "GPU %d%%", gpu_);
            else snprintf(lines[n].text, sizeof lines[n].text, "GPU N/A");
            lines[n++].color = gpu_ >= 95 ? kWarn : gpu_ < 0 ? kDim : kText;
        }
        if (!NVN_FLAG("no_hud_ram")) {
            snprintf(lines[n].text, sizeof lines[n].text, "RAM %.1f GB  %.1f/%.1f", ram_app_, ram_used_, ram_total_);
            lines[n++].color = ram_total_ > 0 && ram_used_ > ram_total_ * 0.9f ? kWarn : kText;
        }
        if (!NVN_FLAG("no_hud_heat")) {  // (Android's thermal headroom: 100 % is where it starts to throttle hard)
            const float h = nvn::thermal_headroom();
            if (h >= 0.0f) snprintf(lines[n].text, sizeof lines[n].text, "HEAT %d%%", static_cast<int>(h * 100.0f + 0.5f));
            else snprintf(lines[n].text, sizeof lines[n].text, "HEAT N/A");
            lines[n++].color = h >= 0.95f ? kBad : h >= 0.8f ? kWarn : h < 0.0f ? kDim : kText;
        }
        if (!NVN_FLAG("no_hud_shaders")) {
            const int building = g_building.load(std::memory_order_relaxed);
            snprintf(lines[n].text, sizeof lines[n].text, "SHADERS %d", std::max(building, 0));
            lines[n++].color = building > 0 ? kWarn : kDim;
        }
        // Debug → Detailed log or GPU breadcrumbs left on: the game runs at a third of its speed or less with either,
        // which looks like stutter and a low frame rate. Said here, so it isn't taken for the port being slow.
        if (n < kStatsLines && (NVN_FLAG("verbose_log") || NVN_FLAG("breadcrumbs") || NVN_FLAG("breadcrumbs_compute") ||
                                NVN_FLAG("breadcrumbs_light"))) {
            snprintf(lines[n].text, sizeof lines[n].text, "DEBUG ON  SLOW");
            lines[n++].color = kBad;
        }
        size_t longest = 0;
        for (int k = 0; k < n; k++) longest = std::max(longest, strlen(lines[k].text));
        const int w = std::min<int>(kStatsW, static_cast<int>(longest) * 12 - 2 + 2 * kStatsPad);
        const int h = std::min<int>(kStatsH, n * kStatsLine + 2 * kStatsPad - 6);
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= w || y >= h) return;
            px[y * static_cast<int>(kStatsW) + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) put(x, y, kBack);
        for (int k = 0; k < n; k++) {
            int x = kStatsPad;
            const int y = kStatsPad + k * kStatsLine;
            for (const char* s = lines[k].text; *s; s++, x += 12) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;  // (a space)
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b))
                            for (int yy = 0; yy < 2; yy++)
                                for (int xx = 0; xx < 2; xx++) put(x + b * 2 + xx, y + r * 2 + yy, lines[k].color);
            }
        }
        *bw = static_cast<uint32_t>(w);
        *bh = static_cast<uint32_t>(h);
    }

    void draw(Rgba* px, bool bgra) {
        const auto put = [&](int x, int y, Rgba c) {
            if (x < 0 || y < 0 || x >= static_cast<int>(kW) || y >= static_cast<int>(kH)) return;
            px[y * kW + x] = bgra ? Rgba{c.b, c.g, c.r, c.a} : c;
        };
        const auto rect = [&](int x0, int y0, int w, int h, Rgba c) {
            for (int y = y0; y < y0 + h; y++)
                for (int x = x0; x < x0 + w; x++) put(x, y, c);
        };
        const auto text = [&](int x, int y, int scale, const char* s, Rgba c) {
            for (; *s; s++, x += 6 * scale) {
                const Glyph* g = nullptr;
                for (const Glyph& f : kFont)
                    if (f.c == *s) g = &f;
                if (!g) continue;
                for (int r = 0; r < 7; r++)
                    for (int b = 0; b < 5; b++)
                        if (g->rows[r] & (0x10 >> b)) rect(x + b * scale, y + r * scale, scale, scale, c);
            }
        };
        rect(0, 0, kW, kH, kBack);

        // El número cambia cuatro veces por segundo, para que se pueda leer.
        if (last_real_ - text_at_ > 250000000 || !text_at_) {
            text_at_ = last_real_;
            shown_ = static_cast<int>(reals_.size() + generated_.size());
        }
        // Verde a su ritmo (30, o 60 con LSFG), amarillo si se queda corto, rojo si va mal.
        const int target = generated_.empty() ? 30 : 60;
        const Rgba color = shown_ >= target - 1 ? kGood : shown_ >= target * 5 / 6 ? kWarn : kBad;
        char line[8];
        snprintf(line, sizeof line, "%d", std::min(shown_, 99));
        const int width = static_cast<int>(strlen(line)) * 6 * 3 - 3;
        text((static_cast<int>(kW) - width) / 2, 3, 3, line, color);
    }

    std::shared_ptr<Buffer> panel_;
    bool panel_bgra_ = false;
    // What the overlay's lines show, read twice a second (sample)
    int64_t sampled_at_ = 0;
    uint64_t cpu_ticks_ = 0;  // the process's user + system time, in clock ticks, at sampled_at_
    int cpu_ = -1, gpu_ = -1, gpu_source_ = -2;  // % (-1: can't be read here); gpu_source_ -2: not looked yet
    float ram_app_ = 0.0f, ram_used_ = 0.0f, ram_total_ = 0.0f;  // GB
    float frame_ms_ = 0.0f, low_fps_ = 0.0f;
    bool failed_ = false;
    bool hidden_ = false, flag_ = false, flag_seen_ = false;  // the counter off; what no_fps said last
    bool panel_open_ = false, touch_seen_ = false, gesture_seen_ = false;
    bool panel_dirty_ = true, panel_drawn_hidden_ = false;  // hay que redibujarlo; el estado con el que se dibujó
    bool choose_folder_ = false;                   // la fila "Cambiar carpeta" (el fichero <datos>/elegir_carpeta)
    bool tv_ = false;                              // the "TV mode" row (tv_mode in mhr.ini)
    bool fg_ = nnsdk::ini::on("lsfg");             // the "Frame gen" row (lsfg in mhr.ini)
    uint32_t panel_w_ = 0, panel_h_ = 0;           // el tamaño con el que se crearon sus buffers
    float panel_from_ = 0.0f;                     // lo que asomaba cuando se abrió o se cerró por última vez
    int64_t panel_at_ = 0, touched_at_ = 0;       // cuándo fue eso, y la última vez que se tocó
    uint32_t touch_handled_ = 0;                  // el toque (NnsdkInput::touch_serial) ya atendido
    uint32_t opened_by_ = 0;                      // el toque que lo abrió deslizando
    uint64_t gesture_ = 0;                        // el último NnsdkInput::hud_gesture atendido
    std::deque<double> intervals_;          // ms entre frames reales (para el 1 % más lento, si vuelve a hacer falta)
    std::deque<int64_t> reals_, generated_;  // en el último segundo
    int64_t last_real_ = 0, text_at_ = 0;
    int shown_ = 0;
};

Hud g_hud;

}  // namespace

void real_frame(int64_t now_ns) { g_hud.real_frame(now_ns); }
void shader_queued() { g_building.fetch_add(1, std::memory_order_relaxed); }
void shader_done() { g_building.fetch_sub(1, std::memory_order_relaxed); }
void generated_frame() { g_hud.generated_frame(); }
void record(Uploads& uploads, VkCommandBuffer cmd, VkImage image, VkExtent2D extent, VkFormat format) {
    g_hud.record(uploads, cmd, image, extent, format);
}

}  // namespace nvn::hud
