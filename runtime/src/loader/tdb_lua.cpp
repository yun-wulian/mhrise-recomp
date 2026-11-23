#include "loader/tdb_lua.h"
#include "loader/application_lua.h"
#include "loader/imports.h"
#include "loader/live_read.h"
#include "loader/lua_explorer.h"
#include "nnsdk/ini.h"
#include "loader/lua_data.h"
#include "loader/lua_frame_gc.h"
#include "loader/lua_mod_hooks.h"
#include "loader/lua_native_hooks.h"
#include "loader/lua_values.h"
#include "loader/mod_ui.h"
#include "loader/nso.h"
#include "loader/tdb71.h"
#include "loader/tls.h"
#include <algorithm>
#include <set>
#include <android/log.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <jni.h>
#include <lua.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace rise;

std::mutex report_mutex;
std::string panel_text;

void write_log(int priority, const char *message) {
  __android_log_write(priority, "RiseLua", message);
  fprintf(priority >= ANDROID_LOG_ERROR ? stderr : stdout, "[RiseLua] %s\n",
          message);
  std::lock_guard lock(report_mutex);
  panel_text.append(message).append("\n");
  if (panel_text.size() > 8192)
    panel_text.erase(0, panel_text.size() - 8192);
}
void report(const char *message, bool error = false) {
  write_log(error ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, message);
}
void push_log_api(lua_State *L) {
  lua_newtable(L);
  for (auto [name, level] : {std::pair{"debug", ANDROID_LOG_DEBUG},
                             std::pair{"info", ANDROID_LOG_INFO},
                             std::pair{"warn", ANDROID_LOG_WARN},
                             std::pair{"error", ANDROID_LOG_ERROR}}) {
    lua_pushinteger(L, level);
    lua_pushcclosure(
        L,
        [](lua_State *s) {
          write_log(lua_tointeger(s, lua_upvalueindex(1)),
                    luaL_checkstring(s, 1));
          return 0;
        },
        1);
    lua_setfield(L, -2, name);
  }
  lua_setglobal(L, "log");
}

Tdb &db(lua_State *L) {
  return *static_cast<Tdb *>(lua_touserdata(L, lua_upvalueindex(1)));
}
int get_name(lua_State *L) {
  lua_pushvalue(L, lua_upvalueindex(1));
  return 1;
}
void named_object(lua_State *L, std::string_view name, const char *getter) {
  lua_newtable(L);
  lua_pushlstring(L, name.data(), name.size());
  lua_pushcclosure(L, get_name, 1);
  lua_setfield(L, -2, getter);
}
char type_id_key;
void push_type(lua_State *L, Tdb &t, uint32_t id);
void bind(lua_State *L, const char *name, lua_CFunction func, Tdb &t,
          uint32_t id) {
  lua_pushlightuserdata(L, &t);
  lua_pushinteger(L, id);
  lua_pushcclosure(L, func, 2);
  lua_setfield(L, -2, name);
}
uint32_t bound_id(lua_State *L) {
  return static_cast<uint32_t>(lua_tointeger(L, lua_upvalueindex(2)));
}
int type_result(lua_State *L) {
  try {
    push_type(L, db(L), bound_id(L));
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int constant(lua_State *L) {
  lua_pushvalue(L, lua_upvalueindex(1));
  return 1;
}
void getter(lua_State *L, const char *label) {
  lua_pushcclosure(L, constant, 1);
  lua_setfield(L, -2, label);
}
int param_types(lua_State *L) {
  try {
    auto &t = db(L);
    size_t sig = t.signature_offset(bound_id(L));
    auto count = read<uint16_t>(t.byte_pool, sig);
    lua_createtable(L, count, 0);
    for (unsigned i = 1; i <= count; ++i) {
      push_type(L, t, t.parameter_type(sig, i));
      lua_rawseti(L, -2, i);
    }
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
void push_method(lua_State *L, Tdb &t, uint32_t id) {
  auto declaring = read<uint32_t>(t.methods, id * 12ull) & id_mask;
  named_object(L, t.method(id, declaring), "get_name");
  mod_lua::tag_method(L, id);
  bind(L, "get_function", rise::descriptor_function, t, id);
  bind(L, "call", rise::invoke_descriptor, t, id);
  lua_newtable(L);
  lua_getfield(L, -2, "call");
  lua_setfield(L, -2, "__call");
  lua_setmetatable(L, -2);
  auto sig = t.signature_offset(id);
  lua_pushinteger(L, read<uint16_t>(t.byte_pool, sig));
  getter(L, "get_num_params");
  bind(L, "get_param_types", param_types, t, id);
  bind(
      L, "get_param_names",
      [](lua_State *s) {
        auto &t = db(s);
        auto sig = t.signature_offset(bound_id(s));
        auto count = read<uint16_t>(t.byte_pool, sig);
        lua_createtable(s, count, 0);
        for (unsigned k = 1; k <= count; ++k) {
          auto p = t.parameter_row(sig, k);
          auto name =
              t.name(read<uint32_t>(t.parameters, 12ull * p + 4) & 0x3fffffff);
          lua_pushlstring(s, name.data(), name.size());
          lua_rawseti(s, -2, k);
        }
        return 1;
      },
      t, id);
  bind(L, "get_return_type", type_result, t, t.parameter_type(sig, 0));
  bind(L, "get_declaring_type", type_result, t, declaring);
  lua_pushboolean(L, read<uint16_t>(t.method_impl(id), 4) & 0x10);
  getter(L, "is_static");
}
int get_method(lua_State *L) {
  const char *name = luaL_checkstring(L, 2);
  try {
    auto &t = db(L);
    auto id = t.lookup_method(bound_id(L), name);
    if (id)
      push_method(L, t, id);
    else
      lua_pushnil(L);
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int get_methods(lua_State *L) {
  try {
    auto &t = db(L);
    auto [first, count] = t.method_range(bound_id(L));
    lua_createtable(L, count, 0);
    for (unsigned k = 0; k < count; ++k) {
      push_method(L, t, first + k);
      lua_rawseti(L, -2, k + 1);
    }
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
template <bool list> int fields(lua_State *L) {
  const char *wanted = list ? nullptr : luaL_checkstring(L, 2);
  try {
    auto &t = db(L);
    auto owner = bound_id(L);
    auto [first, count] = t.field_range(owner);
    if (list)
      lua_createtable(L, count, 0);
    for (size_t i = 0; i < count; ++i) {
      auto f = t.field(first + i, owner);
      if (wanted && f.name != wanted)
        continue;
      named_object(L, f.name, "get_name");
      bind(L, "get_data", rise::read_descriptor_field, t, first + i);
      bind(L, "set_data", rise::write_descriptor_field, t, first + i);
      bind(L, "get_type", type_result, t, f.type);
      bind(L, "get_declaring_type", type_result, t, owner);
      lua_pushboolean(L, f.flags & 0x10);
      getter(L, "is_static");
      lua_pushboolean(L, f.flags & 0x40);
      getter(L, "is_literal");
      lua_pushinteger(L, f.flags);
      getter(L, "get_flags");
      lua_pushinteger(L, f.offset);
      getter(L, "get_offset_from_fieldptr");
      bind(L, "get_offset_from_base", [](lua_State *s) {
        auto &t = db(s);
        auto index = bound_id(s);
        auto declaring = read<uint32_t>(t.fields, 8ull * index) & id_mask;
        lua_pushinteger(s, t.field_offset_from_base(t.field(index, declaring)));
        return 1;
      }, t, first + i);
      if (!list)
        return 1;
      lua_rawseti(L, -2, i + 1);
    }
    if (!list)
      lua_pushnil(L);
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
void push_type(lua_State *L, Tdb &t, uint32_t id) {
  if (!id) {
    lua_pushnil(L);
    return;
  }
  static char type_cache;
  lua_rawgetp(L, LUA_REGISTRYINDEX, &type_cache);
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_rawsetp(L, LUA_REGISTRYINDEX, &type_cache);
  }
  lua_rawgeti(L, -1, id);
  if (!lua_isnil(L, -1)) {
    lua_remove(L, -2);
    return;
  }
  lua_pop(L, 1);
  named_object(L, t.full_name(id), "get_full_name");
  lua_pushinteger(L, id);
  lua_rawsetp(L, -2, &type_id_key);
  auto info = t.impl(size_t(id) * 72);
  for (auto [label, offset] :
       {std::pair{"get_name", 0}, std::pair{"get_namespace", 4}}) {
    auto name = t.name(read<uint32_t>(info, offset));
    lua_pushlstring(L, name.data(), name.size());
    getter(L, label);
  }
  bind(L, "get_parent_type", type_result, t, t.parent(id));
  bind(L, "get_methods", get_methods, t, id);
  bind(L, "get_runtime_type", rise::descriptor_runtime_type, t, id);
  bind(L, "create_instance", rise::descriptor_create_instance, t, id);
  lua_pushinteger(L, read<uint32_t>(t.types, id * 72ull + 20));
  getter(L, "get_size");
  lua_pushinteger(L, read<uint32_t>(info, 8));
  getter(L, "get_valuetype_size");
  lua_pushinteger(L, read<uint32_t>(t.types, id * 72ull + 24));
  getter(L, "get_fqn_hash");
  lua_pushinteger(L, read<uint32_t>(t.types, id * 72ull + 28));
  getter(L, "get_crc_hash");
  lua_pushboolean(L, t.parent(id) == t.find("System.Enum"));
  getter(L, "is_enum");
  auto inherits = [&](uint32_t base) {
    for (auto k = id; k; k = t.parent(k))
      if (k == base)
        return true;
    return false;
  };
  lua_pushboolean(L, inherits(t.find("System.ValueType")));
  getter(L, "is_value_type");
  lua_pushboolean(L, inherits(t.find("System.Array")));
  getter(L, "is_array");
  bind(
      L, "is_a",
      [](lua_State *s) {
        auto &t = db(s);
        if (lua_istable(s, 2)) {
          lua_getfield(s, 2, "get_full_name");
          luaL_checktype(s, -1, LUA_TFUNCTION);
          lua_call(s, 0, 1);
        }
        auto base = t.find(luaL_checkstring(s, -1));
        for (auto k = bound_id(s); k; k = t.parent(k))
          if (k == base) {
            lua_pushboolean(s, 1);
            return 1;
          }
        lua_pushboolean(s, 0);
        return 1;
      },
      t, id);
  bind(L, "get_method", get_method, t, id);
  bind(L, "get_field", fields<false>, t, id);
  bind(L, "get_fields", fields<true>, t, id);
  lua_pushvalue(L, -1);
  lua_rawseti(L, -3, id);
  lua_remove(L, -2);
}
int find_type(lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  try {
    auto &t = db(L);
    push_type(L, t, t.find(name));
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}

// Switch 16.0.1 main. Verified on-device against the live TDB and the managed
// vtable entries for PlayerManager.awake/start/onDestroy/update.
constexpr size_t update_rva = 0x5b79bbc, instance_rva = 0x2629f0,
                 find_rva = 0x2b9b9c;
constexpr size_t update_method = 225661, instance_method = 222817,
                 find_method = 225669;
struct PlayerProbe {
  const Module &main;
  Tdb tdb;
  fs::path script, request;
  uint32_t manager_id, player_id;
  std::atomic<unsigned> pending{
      0}; // bit 0: player probe; bit 1: yun data probe

  PlayerProbe(const Module &m, const fs::path &dir)
      : main(m), tdb(m), script(dir / "player_probe.lua"),
        request(dir / "player.request"),
        manager_id(tdb.find("snow.player.PlayerManager")),
        player_id(tdb.find("snow.player.PlayerBase")) {
    auto parent = parent_of(manager_id);
    if (!manager_id || !player_id ||
        tdb.method(update_method, manager_id) != "update" ||
        tdb.method(instance_method, parent) != "get_Instance" ||
        tdb.method(find_method, manager_id) != "findMasterPlayer")
      throw std::runtime_error(
          "Player probe metadata does not match the build profile");
  }
  uint32_t parent_of(uint32_t id) const {
    return (read<uint64_t>(tdb.types, size_t(id) * 72) >> 19) & id_mask;
  }
  uint32_t object_type(uintptr_t object) const {
    auto info = *reinterpret_cast<const uintptr_t *>(object);
    if (!info)
      throw std::runtime_error("Managed object has no class info");
    auto type = *reinterpret_cast<const uintptr_t *>(info);
    auto start = reinterpret_cast<uintptr_t>(tdb.types.data());
    if (type < start || type - start >= tdb.types.size() || (type - start) % 72)
      throw std::runtime_error("Object type is outside the active TDB");
    return (type - start) / 72;
  }
  std::string full_name(uint32_t id) const {
    auto impl = tdb.impl(size_t(id) * 72);
    return std::string(tdb.name(read<uint32_t>(impl, 4))) + "." +
           std::string(tdb.name(read<uint32_t>(impl, 0)));
  }
  bool is_player(uint32_t id) const {
    for (size_t depth = 0; id && depth < tdb.types.size() / 72; ++depth) {
      if (id == player_id)
        return true;
      id = parent_of(id);
    }
    return false;
  }
  intptr_t encoded(size_t index) const {
    return read<int32_t>(tdb.methods, index * 12 + 8);
  }
  void validate_live_methods() const {
    const auto anchor =
        static_cast<intptr_t>(update_rva) - encoded(update_method);
    if (!encoded(update_method) ||
        anchor + encoded(instance_method) != instance_rva ||
        anchor + encoded(find_method) != find_rva)
      throw std::runtime_error(
          "Live TDB method offsets do not match the Switch build");
  }
};
std::unique_ptr<PlayerProbe> player_probe;
std::unique_ptr<Tdb> resident_metadata;
mod_lua::FrameGc resident_gc;
std::unique_ptr<lua_State, decltype(&lua_close)> resident_vm(nullptr,
                                                             lua_close);
std::unique_ptr<rise::ReadFrame> resident_frame;

// Windows mods sometimes require a module with different casing from its
// filename. Keep the ordinary Lua searcher first so exact names always win.
int pc_module_searcher(lua_State *L) {
  std::string name = luaL_checkstring(L, 1);
  std::replace(name.begin(), name.end(), '.', '/');
  std::replace(name.begin(), name.end(), '\\', '/');
  fs::path rel(name), root = lua_tostring(L, lua_upvalueindex(1));
  if (rel.empty() || rel.is_absolute())
    return 0;
  auto fold = [](std::string text) {
    for (char &c : text)
      if (c >= 'A' && c <= 'Z')
        c += 32;
    return text;
  };
  try {
    for (auto candidate : {fs::path(name + ".lua"), rel / "init.lua"}) {
      fs::path path = root;
      for (const auto &component : candidate) {
        if (fs::exists(path / component)) {
          path /= component;
          continue;
        }
        if (!fs::is_directory(path)) {
          path.clear();
          break;
        }
        std::optional<fs::path> match;
        auto key = fold(component.string());
        for (const auto &entry : fs::directory_iterator(path))
          if (fold(entry.path().filename().string()) == key) {
            if (match)
              throw std::runtime_error("Ambiguous module filename: " + name);
            match = entry.path();
          }
        if (!match) {
          path.clear();
          break;
        }
        path = *match;
      }
      if (!path.empty() && fs::is_regular_file(path)) {
        if (luaL_loadfilex(L, path.c_str(), "t") != LUA_OK)
          return lua_error(L);
        lua_pushstring(L, path.c_str());
        return 2;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}

std::vector<fs::path> resident_roots, launch_mixed_roots;
enum class RuntimeStatus : jint { Inactive, WaitingForPlayer, Running };
std::atomic<RuntimeStatus> runtime_status{RuntimeStatus::Inactive};

std::vector<fs::path> mixed_roots(const std::vector<fs::path> &roots) {
  std::vector<fs::path> mixed;
  for (size_t i = 1; i < roots.size(); ++i) { // entry zero is the private script directory
    const auto dir = roots[i].parent_path();
    if (fs::is_directory(dir / "romfs") || fs::is_directory(dir / "exefs"))
      mixed.push_back(roots[i]);
  }
  return mixed;
}

bool load_scripts(lua_State *L, const fs::path &directory) {
  std::vector<fs::path> scripts;
  auto roots = directory.filename() == "autorun"
                   ? resident_roots
                   : std::vector<fs::path>{directory.parent_path()};
  std::set<std::string> seen;
  for (auto &root : roots) {
    auto dir = root / directory.filename();
    if (!fs::is_directory(dir))
      continue;
    for (const auto &entry : fs::directory_iterator(dir)) {
      auto path = entry.path();
      if (entry.is_regular_file() && path.extension() == ".lua" &&
          seen.insert(path.filename().string()).second)
        scripts.push_back(path);
    }
  }
  std::sort(scripts.begin(), scripts.end(), [](const auto &a, const auto &b) {
    return a.filename() < b.filename();
  });
  bool ok = true;
  for (const auto &path : scripts) {
    auto previous = mod_lua::checkpoint();
    auto previous_events = mod_lua::event_checkpoint(L);
    mod_lua::set_loading_script(path.stem().string());
    bool failed = luaL_loadfilex(L, path.c_str(), "t") || lua_pcall(L, 0, 0, 0);
    mod_lua::set_loading_script({});
    if (failed) {
      const char *error = lua_tostring(L, -1);
      auto message = path.filename().string() + ": " + (error ? error : "non-string Lua error");
      report(message.c_str(), true);
      mod_lua::script_error(message);
      lua_pop(L, 1);
      mod_lua::rollback(L, previous);
      mod_lua::rollback_events(L, previous_events);
      ok = false;
    } else {
      report(("Loaded " + path.filename().string()).c_str());
    }
  }
  return ok;
}
bool initialize_resident(const Module &main, const fs::path &directory,
                         bool run = true) {
  auto &tdb = *resident_metadata;
  mod_lua::clear_script_errors();
  resident_vm.reset(luaL_newstate());
  auto *L = resident_vm.get();
  if (!L)
    throw std::runtime_error("Could not create Lua state");
  luaL_openlibs(L);
  mod_lua::open_events(L);
  mod_lua::open_imgui(L);
  mod_lua::open_values(L);
  mod_lua::open_data(L, directory.parent_path(), resident_roots);
  lua_getglobal(L, "package");
  const auto &roots = resident_roots;
  std::string search;
  for (const auto &root : roots) {
    auto dir = (root / "autorun").string();
    if (!search.empty())
      search += ";";
    search += dir + "/?.lua;" + dir + "/?/init.lua";
  }
  lua_pushstring(L, search.c_str());
  lua_setfield(L, -2, "path");
  lua_getfield(L, -1, "searchers");
  for (const auto &root : roots) {
    lua_pushstring(L, (root / "autorun").c_str());
    lua_pushcclosure(L, pc_module_searcher, 1);
    lua_rawseti(L, -2, lua_rawlen(L, -2) + 1);
  }
  lua_pop(L, 2);
  lua_newtable(L);
  lua_pushlightuserdata(L, &tdb);
  lua_pushcclosure(L, find_type, 1);
  lua_setfield(L, -2, "find_type_definition");
  lua_pushcfunction(L, [](lua_State *s) {
    lua_pushinteger(s, 71);
    return 1;
  });
  lua_setfield(L, -2, "get_tdb_version");
  lua_pushcfunction(L, [](lua_State *s) {
    luaL_checktype(s, 1, LUA_TSTRING);
    lua_pushliteral(s, "snow.");
    lua_pushvalue(s, 1);
    lua_concat(s, 2);
    return 1;
  });
  lua_setfield(L, -2, "game_namespace");
  mod_lua::install_api(L, tdb, main);
  lua_setglobal(L, "sdk");
  push_log_api(L);
  if (resident_frame)
    open_live_read(L, *resident_frame);
  mod_lua::select_event_state(L, &resident_gc);
  mod_lua::watch_missing_calls(L);
  return !run || load_scripts(L, directory);
}

} // namespace

void mod_lua::push_type_definition(lua_State *L, rise::Tdb &t, uint32_t id) {
  push_type(L, t, id);
}
uint32_t mod_lua::type_definition_id(lua_State *L, int index) {
  luaL_checktype(L, index, LUA_TTABLE);
  lua_rawgetp(L, index, &type_id_key);
  if (!lua_isinteger(L, -1))
    luaL_argerror(L, index, "expected an RETypeDefinition");
  auto id = static_cast<uint32_t>(lua_tointeger(L, -1));
  lua_pop(L, 1);
  return id;
}
void mod_lua::log_message(int priority, const char *message) {
  write_log(priority, message);
}

size_t prepare_player_probe(const Module &main, const fs::path &directory) {
  constexpr uint8_t build_id[32] = {0x55, 0xd5, 0x0c, 0xa1, 0x80, 0x5e, 0x9c,
                                    0x5b, 0xba, 0x50, 0x8d, 0x60, 0x1a, 0xdd,
                                    0x85, 0x2f, 0x25, 0x5a, 0x32, 0xba};
  try {
    if (!resident_vm)
      return 0;
    if (memcmp(main.build_id, build_id, sizeof(build_id)) != 0)
      throw std::runtime_error("Unsupported main build ID for player probe");
    const Bytes text(main.base + main.seg_off[0], main.seg_size[0]);
    if (main.seg_off[0] != 0 || read<uint32_t>(text, 0x6b61f2c) != 0xa9ba6ffc ||
        read<uint32_t>(text, 0x6b621cc) != 0xf8687928 ||
        read<uint32_t>(text, update_rva) != 0xd10103ff ||
        read<uint32_t>(text, instance_rva) != 0xf00920a8 ||
        read<uint32_t>(text, find_rva) != 0xf9405428)
      throw std::runtime_error(
          "Player probe entry points differ from the verified NS build");
    player_probe = std::make_unique<PlayerProbe>(main, directory);
    report("PlayerManager.update probe ready");
    return update_rva;
  } catch (const std::exception &e) {
    report(e.what(), true);
    return 0;
  }
}
void poll_player_request() {
  if (!player_probe)
    return;
  if (unlink(player_probe->request.c_str()) == 0)
    player_probe->pending.fetch_or(1);
  if (unlink((player_probe->request.parent_path() / "yun.request").c_str()) ==
      0)
    player_probe->pending.fetch_or(2);
  if (unlink(
          (player_probe->request.parent_path() / "reload.request").c_str()) ==
      0)
    player_probe->pending.fetch_or(4);
  if (unlink(
          (player_probe->request.parent_path() / "config.request").c_str()) ==
      0)
    player_probe->pending.fetch_or(8);
}
void on_player_update(uintptr_t context, uintptr_t manager) {
  auto &p = *player_probe;
  static std::once_flag attach;
  std::call_once(attach, [&] {
    try {
      std::lock_guard lock(mod_lua::lua_mutex());
      p.validate_live_methods();
      if (rise::current_vm_context(p.main) != context)
        throw std::runtime_error("Engine thread context mismatch");
      resident_frame = std::make_unique<rise::ReadFrame>(rise::ReadFrame{
          p.main, *resident_metadata, 0, 0,
          p.main.addr(update_rva) - p.encoded(update_method), true});
      open_live_read(resident_vm.get(), *resident_frame);
      mod_lua::install_app_entries(p.main, *resident_metadata);
      load_scripts(resident_vm.get(), p.script.parent_path() / "autorun");
      auto live = p.script.parent_path() / "live";
      if (fs::exists(live))
        load_scripts(resident_vm.get(), live);
      mod_lua::load_object_explorer(resident_vm.get(), *resident_metadata);
      runtime_status.store(RuntimeStatus::Running);
    } catch (const std::exception &e) {
      report(e.what(), true);
    }
  });
  rise::drain_managed_releases();
  if (!p.pending.load(std::memory_order_relaxed))
    return;
  unsigned pending = p.pending.exchange(0);
  if (!pending)
    return;
  if (pending & 8) {
    mod_lua::save_config();
    pending &= ~8u;
    if (!pending)
      return;
  }
  if (pending & 4) {
    try {
      std::lock_guard lock(mod_lua::lua_mutex());
      if (!resident_frame)
        throw std::runtime_error("Live runtime is not initialized");
      // A reflected method can re-enter PlayerManager.update from a Lua hook.
      // Leave the request pending until the previous VM is off the call stack.
      lua_Debug activation{};
      if (mod_lua::native_hooks_in_flight() ||
          (resident_vm && lua_getstack(resident_vm.get(), 0, &activation))) {
        p.pending.fetch_or(pending);
        return;
      }
      auto dir = p.script.parent_path();
      auto roots = mod_lua::framework_roots(dir);
      // A pure-script reload may have been queued before a mixed mod was installed or toggled.
      // Keep the running VM paired with the resource set loaded by the game at startup.
      if (mixed_roots(roots) != launch_mixed_roots)
        throw std::runtime_error("Restart the game to apply resource mod changes");
      mod_lua::reset_callbacks();
      mod_lua::select_event_state(nullptr, nullptr);
      mod_lua::rollback(resident_vm.get(), 0);
      runtime_status.store(RuntimeStatus::Inactive);
      resident_vm.reset();
      resident_roots = std::move(roots);
      bool ok = initialize_resident(p.main, dir / "autorun");
      if (fs::exists(dir / "live"))
        ok = load_scripts(resident_vm.get(), dir / "live") && ok;
      mod_lua::load_object_explorer(resident_vm.get(), *resident_metadata);
      runtime_status.store(RuntimeStatus::Running);
      report(ok ? "Lua reload complete"
                : "Lua reload finished with script errors",
             !ok);
    } catch (const std::exception &e) {
      report(e.what(), true);
    }
    pending &= ~4u;
    if (!pending)
      return;
  }
  const bool yun = (pending & 2) != 0;
  if (yun && (pending & 1))
    p.pending.fetch_or(1);
  try {
    if (!context || !manager)
      throw std::runtime_error("Missing update receiver or VM context");
    p.validate_live_methods();
    char thread[32] = {};
    pthread_getname_np(pthread_self(), thread, sizeof(thread));
    report(("PlayerManager.update on " + std::string(thread) +
            " tid=" + std::to_string(gettid()))
               .c_str());
    std::unique_ptr<lua_State, decltype(&lua_close)> state(luaL_newstate(),
                                                           lua_close);
    auto *L = state.get();
    if (!L)
      throw std::runtime_error("Could not create player probe Lua state");
    luaL_openlibs(L);
    rise::ReadFrame frame{p.main, p.tdb, context, manager,
                          p.main.addr(update_rva) - p.encoded(update_method)};
    open_live_read(L, frame);
    mod_lua::open_values(L);
    auto autorun = (p.script.parent_path() / "autorun").string();
    lua_getglobal(L, "package");
    lua_pushstring(L, (autorun + "/?.lua;" + autorun + "/?/init.lua").c_str());
    lua_setfield(L, -2, "path");
    lua_pop(L, 1);
    push_log_api(L);
    auto script =
        yun ? p.script.parent_path() / "yun_data_probe.lua" : p.script;
    if (luaL_loadfilex(L, script.c_str(), "t") || lua_pcall(L, 0, 0, 0)) {
      const char *err = lua_tostring(L, -1);
      report(err ? err : "Non-string Lua error in player probe", true);
    }
  } catch (const std::exception &e) {
    report(e.what(), true);
  }
}

// For the MOD UI button: 0 no Lua runtime (no scripts installed), 1 scripts loaded but not running yet (they start
// with the game's first frame in a village or quest), 2 running
extern "C" JNIEXPORT jint JNICALL Java_io_github_iamwilmar_mhriseport_ModUi_nativeRuntime(JNIEnv *, jclass) {
  return static_cast<jint>(runtime_status.load());
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_github_iamwilmar_mhriseport_SettingsMenu_nativeLuaReport(JNIEnv *env,
                                                                 jclass) {
  std::string text;
  {
    std::unique_lock lua_lock(mod_lua::lua_mutex(), std::try_to_lock);
    if (lua_lock.owns_lock()) {
      const auto &s = resident_gc.stats();
      if (s.heap_bytes) {
        char line[200];
        snprintf(line, sizeof(line), "Lua GC: %.1f MiB, last active slice %lld/%d us, peak step %lld us, %llu cycles\n",
                 double(s.heap_bytes) / 1048576, (long long)s.slice_us, s.budget_us,
                 (long long)s.peak_step_us, (unsigned long long)s.cycles);
        text = line;
      }
    }
  }
  {
    std::lock_guard lock(report_mutex);
    text += panel_text;
  }
  jbyteArray result = env->NewByteArray(static_cast<jsize>(text.size()));
  if (result)
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(text.size()),
                            reinterpret_cast<const jbyte *>(text.data()));
  return result;
}

bool run_tdb_autorun(const Module &main, const fs::path &directory) {
  try {
    auto roots = mod_lua::framework_roots(directory.parent_path());
    if (!nnsdk::ini::on("object_explorer") && std::none_of(roots.begin(), roots.end(), [](const auto &r) {
          return fs::is_directory(r / "autorun");
        })) {
      report("No autorun directory");
      return false;
    }
    if (resident_vm)
      throw std::runtime_error("Autorun is already initialized");
    launch_mixed_roots = mixed_roots(roots);
    resident_roots = std::move(roots);
    resident_metadata = std::make_unique<Tdb>(main);
    bool ok = initialize_resident(main, directory, getenv("NIER_INTERNO") == nullptr);
    runtime_status.store(RuntimeStatus::WaitingForPlayer);
    return ok;
  } catch (const std::exception &e) {
    report(e.what(), true);
    return false;
  }
}
