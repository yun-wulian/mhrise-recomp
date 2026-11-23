#include "loader/lua_native_hooks.h"
#include "loader/application_lua.h"
#include "loader/live_read.h"
#include "loader/mod_ui.h"
#include "loader/lua_mod_hooks.h"
#include "loader/lua_values.h"
#include <algorithm>
#include <android/log.h>
#include <atomic>
#include <cstring>
#include <lua.hpp>
#include <memory>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mod_lua {
namespace {
using namespace rise;
enum class Kind {
  Void,
  Integer,
  Float,
  Double,
  Reference,
  Value,
  ByRef,
  Context
};
struct Type {
  Kind kind;
  uint32_t id;
  uint32_t size;
  bool operator==(const Type &b) const {
    return kind == b.kind && size == b.size &&
           ((kind != Kind::Value && kind != Kind::ByRef) || id == b.id);
  }
};
struct Argument {
  Type type;
  unsigned location;
  unsigned offset;
};
struct Layout {
  std::vector<Argument> args;
  Type result;
  size_t stack_bytes{};
  bool operator==(const Layout &b) const {
    if (!(result == b.result) || args.size() != b.args.size() ||
        stack_bytes != b.stack_bytes)
      return false;
    for (size_t i = 0; i < args.size(); ++i)
      if (!(args[i].type == b.args[i].type) ||
          args[i].location != b.args[i].location ||
          args[i].offset != b.args[i].offset)
        return false;
    return true;
  }
};
struct Subscriber {
  uint64_t serial, epoch;
  uint32_t method;
  uint32_t owner;
  Layout layout;
  int pre, post;
  std::atomic<bool> enabled{true};
};
struct Site {
  NativePatch
      patch; // Must be the first member: the generated entry loads its address.
  size_t stack_bytes{};
  std::vector<uint32_t> owners;
  std::vector<std::shared_ptr<Subscriber>> subscribers;
  std::atomic<size_t> active{0};
};
static_assert(offsetof(Site, patch) == 0);
std::vector<std::unique_ptr<Site>> sites;
std::unordered_map<uintptr_t, Site *> by_address;
std::unordered_map<uint64_t, std::shared_ptr<Subscriber>> subscribers;
const Module *module = nullptr;
Tdb *db = nullptr;
lua_State *vm = nullptr;
uint64_t epoch = 0, serial = 0;
// Snapshot before installing the first hook. Lua reloads keep the native sites.
std::vector<uint32_t> destinations;
std::vector<std::pair<uint32_t, uint32_t>> entries;
thread_local unsigned native_depth = 0;
bool derives(uint32_t id, uint32_t parent) {
  for (; id; id = db->parent(id))
    if (id == parent)
      return true;
  return false;
}
Type type(uint32_t id) {
  auto name = db->full_name(id);
  if (name == "System.Void")
    return {Kind::Void, id, 0};
  if (derives(id, db->find("System.Enum"))) {
    auto [first, count] = db->field_range(id);
    if (!count || db->field(first, id).name != "value__")
      throw std::runtime_error("Enum has no underlying type");
    auto t = type(db->field(first, id).type);
    t.id = id;
    return t;
  }
  if (name == "System.Single")
    return {Kind::Float, id, 4};
  if (name == "System.Double")
    return {Kind::Double, id, 8};
  for (auto [n, width] : {std::pair{"Boolean", 1u},
                          {"Byte", 1u},
                          {"SByte", 1u},
                          {"Int16", 2u},
                          {"UInt16", 2u},
                          {"Char", 2u},
                          {"Int32", 4u},
                          {"UInt32", 4u},
                          {"Int64", 8u},
                          {"UInt64", 8u},
                          {"IntPtr", 8u},
                          {"UIntPtr", 8u}})
    if (name == std::string("System.") + n)
      return {Kind::Integer, id, width};
  if (!derives(id, db->find("System.ValueType")))
    return {Kind::Reference, id, 8};
  auto bytes = read<uint32_t>(db->impl(72ull * id), 8);
  if (!bytes)
    throw std::runtime_error("Value type has no concrete size: " + name);
  // RE's generated managed C++ value wrappers are non-trivial for calls:
  // parameters are indirect, and aggregate returns use x8.
  return {Kind::Value, id, bytes};
}
Layout describe(uint32_t method) {
  auto sig = db->signature_offset(method);
  auto count = read<uint16_t>(db->byte_pool, sig);
  Layout out;
  out.result = type(db->parameter_type(sig, 0));
  if (!db->plain_parameter(sig, 0))
    out.result = {Kind::ByRef, out.result.id, 8};
  unsigned gp = 0, fp = 0, stack = 0;
  auto add = [&](Type t) {
    bool floating = t.kind == Kind::Float || t.kind == Kind::Double;
    unsigned &reg = floating ? fp : gp;
    if (reg < 8)
      out.args.push_back({t, floating ? 1u : 0u, reg++});
    else {
      out.args.push_back({t, 2, stack});
      stack += 8;
    }
  };
  add({Kind::Context, 0, 8});
  auto owner = read<uint32_t>(db->methods, 12ull * method) & id_mask;
  if (!(read<uint16_t>(db->method_impl(method), 4) & 0x10))
    add({Kind::Reference, owner, 8});
  for (unsigned i = 1; i <= count; ++i) {
    auto t = type(db->parameter_type(sig, i));
    if (!db->plain_parameter(sig, i))
      t = {Kind::ByRef, t.id, 8};
    if (t.kind == Kind::Void)
      throw std::runtime_error("Void parameter");
    add(t);
  }
  out.stack_bytes = (stack + 15) & ~size_t(15);
  return out;
}
struct Invocation {
  const Layout &layout() const { return callbacks.front()->layout; }
  uint64_t generation{};
  std::vector<std::shared_ptr<Subscriber>> callbacks;
  std::vector<int> storage;
  std::vector<int> held_objects;
  std::vector<uint8_t> stack;
  std::vector<std::pair<uintptr_t, Type>> buffers;
  std::unordered_map<uintptr_t, uint32_t> objects;
  int args = LUA_NOREF;
  Invocation *parent{};
  int top{};
  ~Invocation();
  bool has_buffer(uintptr_t p, const Type &t) const {
    for (const auto &[address, known] : buffers)
      if (p == address && t.id == known.id && t.kind == known.kind)
        return true;
    return false;
  }
};
thread_local Invocation *current = nullptr;
Invocation::~Invocation() {
  std::lock_guard lock(lua_mutex());
  if (generation != epoch)
    return;
  for (int ref : storage)
    luaL_unref(vm, LUA_REGISTRYINDEX, ref);
  for (int ref : held_objects)
    luaL_unref(vm, LUA_REGISTRYINDEX, ref);
  luaL_unref(vm, LUA_REGISTRYINDEX, args);
}
uint64_t raw_at(const Argument &arg, const NativeFrame &f) {
  uint64_t value{};
  size_t width = (arg.type.kind == Kind::Integer ||
                  arg.type.kind == Kind::Float || arg.type.kind == Kind::Double)
                     ? arg.type.size
                     : 8;
  if (arg.location == 0)
    memcpy(&value, &f.x[arg.offset], width);
  else if (arg.location == 1)
    memcpy(&value, f.v[arg.offset], width);
  else
    memcpy(&value, f.stack_copy + arg.offset, width);
  return value;
}
void put_at(const Argument &arg, NativeFrame &f, uint64_t value) {
  if (arg.location == 0) {
    f.x[arg.offset] = value;
    return;
  }
  if (arg.location == 1) {
    memcpy(f.v[arg.offset], &value, arg.type.size);
    return;
  }
  memcpy(const_cast<uint8_t *>(f.stack_copy) + arg.offset, &value, 8);
}
void push_bits(lua_State *L, uint64_t bits) {
  lua_pushlightuserdata(L, reinterpret_cast<void *>(bits));
}
uint64_t bits(lua_State *L, int index) {
  if (lua_islightuserdata(L, index))
    return reinterpret_cast<uintptr_t>(lua_touserdata(L, index));
  if (lua_isnil(L, index))
    return 0;
  if (lua_isboolean(L, index))
    return lua_toboolean(L, index);
  if (lua_isinteger(L, index))
    return lua_tointeger(L, index);
  if (is_live_object(L, index))
    return live_object_address(L, index);
  throw std::runtime_error(
      "Expected sdk.to_ptr, a hook argument, or managed object");
}
void report_error(Subscriber &s, const std::string &error) {
  if (!s.enabled.exchange(false))
    return;
  auto owner = read<uint32_t>(db->methods, 12ull * s.method) & id_mask;
  auto message = db->full_name(owner) + "." + std::string(db->method(s.method, owner)) + " hook disabled: " + error;
  log_message(ANDROID_LOG_ERROR, message.c_str());
  script_error(message);
}
void execute(Invocation &inv, size_t n, int ref, int argument) {
  auto previous = swap_hook_storage(&inv.storage[n]);
  lua_rawgeti(vm, LUA_REGISTRYINDEX, ref);
  lua_pushvalue(vm, argument);
  int status = lua_pcall(vm, 1, 1, 0);
  swap_hook_storage(previous);
  if (status != LUA_OK)
    throw std::runtime_error(lua_isstring(vm, -1)
                                 ? lua_tostring(vm, -1)
                                 : "Non-string Lua hook error");
}
void build_arguments(Invocation &inv, NativeFrame &frame) {
  const auto &result = inv.layout().result;
  // Win64 exposes the caller's aggregate result buffer before the context.
  // AArch64 keeps it separately in x8.
  bool hidden = result.kind == Kind::Value && result.size != 1 &&
                result.size != 2 && result.size != 4 && result.size != 8;
  if (result.kind == Kind::Value && frame.x[8])
    inv.buffers.emplace_back(frame.x[8], result);
  lua_createtable(vm, inv.layout().args.size() + hidden, 0);
  if (hidden) {
    push_bits(vm, frame.x[8]);
    lua_rawseti(vm, -2, 1);
  }
  for (size_t i = 0; i < inv.layout().args.size(); ++i) {
    const auto &a = inv.layout().args[i];
    auto value = raw_at(a, frame);
    if (a.type.kind == Kind::Reference && value)
      inv.objects.emplace(value, a.type.id);
    if ((a.type.kind == Kind::Value || a.type.kind == Kind::ByRef) && value)
      inv.buffers.emplace_back(value, a.type);
    push_bits(vm, value);
    lua_rawseti(vm, -2, i + 1 + hidden);
  }
  inv.args = luaL_ref(vm, LUA_REGISTRYINDEX);
}
uint64_t changed_argument(Invocation &inv, const Argument &a, int value,
                          uint64_t original) {
  uint64_t v;
  if ((a.type.kind == Kind::Float || a.type.kind == Kind::Double) &&
      lua_type(vm, value) == LUA_TNUMBER) {
    if (a.type.kind == Kind::Float) {
      float f = lua_tonumber(vm, value);
      uint32_t w;
      memcpy(&w, &f, 4);
      v = w;
    } else {
      double d = lua_tonumber(vm, value);
      memcpy(&v, &d, 8);
    }
  } else
    v = bits(vm, value);
  if (v == original)
    return v;
  if (a.type.kind == Kind::Context)
    throw std::runtime_error("Cannot replace the engine VM context");
  if ((a.type.kind == Kind::Value || a.type.kind == Kind::ByRef) &&
      !inv.has_buffer(v, a.type))
    throw std::runtime_error(
        "Replacing an indirect value requires a typed value buffer");
  if (a.type.kind == Kind::Reference && v) {
    if (is_live_object(vm, value)) {
      live_object_address(vm, value, a.type.id);
      inv.objects.emplace(v, a.type.id);
      lua_pushvalue(vm, value);
      inv.held_objects.push_back(luaL_ref(vm, LUA_REGISTRYINDEX));
    } else {
      auto it = inv.objects.find(v);
      if (it == inv.objects.end()) {
        if (!push_cached_live_object(vm, v, a.type.id))
          throw std::runtime_error(
              "Unknown managed object in replacement argument");
      } else
        push_live_object(vm, v, a.type.id);
      inv.held_objects.push_back(luaL_ref(vm, LUA_REGISTRYINDEX));
    }
  }
  return v;
}
bool before(Invocation &inv, NativeFrame &frame) {
  bool skip = false;
  auto count = inv.layout().args.size();
  std::vector<uint64_t> pending(count);
  for (size_t n = 0; n < inv.callbacks.size(); ++n) {
    auto &s = *inv.callbacks[n];
    if (!s.enabled || s.pre == LUA_NOREF)
      continue;
    int top = lua_gettop(vm);
    try {
      lua_rawgeti(vm, LUA_REGISTRYINDEX, inv.args);
      int original = lua_gettop(vm);
      size_t slots = lua_rawlen(vm, original);
      size_t hidden = slots - count;
      lua_createtable(vm, slots, 0);
      int draft = lua_gettop(vm);
      for (size_t i = 1; i <= slots; ++i) {
        lua_rawgeti(vm, original, i);
        lua_rawseti(vm, draft, i);
      }
      execute(inv, n, s.pre, draft);
      auto action = pre_action(vm);
      if (hidden) {
        lua_rawgeti(vm, draft, 1);
        if (bits(vm, -1) != frame.x[8])
          throw std::runtime_error("Cannot redirect caller-owned return storage");
        lua_pop(vm, 1);
      }
      for (size_t i = 0; i < count; ++i) {
        lua_rawgeti(vm, draft, i + 1 + hidden);
        pending[i] =
            changed_argument(inv, inv.layout().args[i], lua_gettop(vm),
                             raw_at(inv.layout().args[i], frame));
        lua_pop(vm, 1);
      }
      for (size_t i = 0; i < count; ++i) {
        put_at(inv.layout().args[i], frame, pending[i]);
        push_bits(vm, pending[i]);
        lua_rawseti(vm, original, i + 1 + hidden);
      }
      skip |= action == PreHookResult::SkipOriginal;
    } catch (const std::exception &e) {
      report_error(s, e.what());
    }
    lua_settop(vm, top);
  }
  return skip;
}
uint64_t result_bits(const Layout &l, const NativeFrame &f) {
  uint64_t value{};
  if (l.result.kind == Kind::Void)
    return 0;
  if (l.result.kind == Kind::Value)
    return f.x[8];
  if (l.result.kind == Kind::Float || l.result.kind == Kind::Double) {
    memcpy(&value, f.result_v[0], l.result.size);
    return value;
  }
  if (l.result.kind == Kind::Integer)
    memcpy(&value, &f.result_x[0], l.result.size);
  else
    value = f.result_x[0];
  return value;
}
void after(Invocation &inv, NativeFrame &frame) {
  const auto &t = inv.layout().result;
  uint64_t result = result_bits(inv.layout(), frame);
  if (t.kind == Kind::Reference && result)
    inv.objects.emplace(result, t.id);
  for (size_t n = inv.callbacks.size(); n > 0; --n) {
    auto &s = *inv.callbacks[n - 1];
    if (!s.enabled || s.post == LUA_NOREF)
      continue;
    int top = lua_gettop(vm);
    try {
      if (t.kind == Kind::Void)
        lua_pushnil(vm);
      else
        push_bits(vm, result);
      execute(inv, n - 1, s.post, top + 1);
      if (!lua_isnil(vm, -1)) {
        auto replacement =
            changed_argument(inv, Argument{t, 0, 0}, lua_gettop(vm), result);
        if (t.kind == Kind::Void && replacement)
          throw std::runtime_error("Nonzero result for void method");
        if (t.kind == Kind::Value && result != replacement) {
          if (!result)
            throw std::runtime_error("Missing aggregate result buffer");
          memmove(reinterpret_cast<void *>(result),
                  reinterpret_cast<const void *>(replacement), t.size);
        } else {
          result = replacement;
        }
      }
    } catch (const std::exception &e) {
      report_error(s, e.what());
    }
    lua_settop(vm, top);
  }
  if (t.kind == Kind::Float || t.kind == Kind::Double)
    memcpy(frame.result_v[0], &result, t.size);
  else if (t.kind != Kind::Value)
    frame.result_x[0] = result;
}
class ActiveCall {
  Invocation *prior;

public:
  ActiveCall(Invocation &inv) : prior(current) {
    inv.parent = prior;
    current = &inv;
  }
  ~ActiveCall() { current = prior; }
};
void index_entries(uintptr_t encoded_base) {
  if (!destinations.empty())
    return;
  const auto text = module->addr(module->seg_off[0]);
  const auto bytes = module->seg_size[0];
  const auto count = db->methods.size() / 12;
  std::vector<uint32_t> targets(count, UINT32_MAX);
  std::vector<std::pair<uint32_t, uint32_t>> ordered;
  ordered.reserve(count);
  for (uint32_t k = 1; k < count; ++k) {
    auto offset = read<int32_t>(db->methods, 12ull * k + 8);
    auto raw = encoded_base + offset;
    if (!offset || raw - text >= bytes || raw % 4)
      continue;
    auto dest = raw;
    if ((*(const uint32_t *)raw >> 26) == 5) {
      try {
        dest = follow_native_entry(*module, raw, true);
      } catch (const std::runtime_error &) {
        continue; // A cyclic entry has no callable destination.
      }
    }
    targets[k] = dest - text;
    ordered.emplace_back(targets[k], k);
  }
  std::sort(ordered.begin(), ordered.end());
  destinations.swap(targets);
  entries.swap(ordered);
}
void describe_site(Site &site, uint32_t method, const Layout &layout,
                   uintptr_t address) {
  site.stack_bytes = layout.stack_bytes;
  auto dest = destinations.at(method);
  auto begin = std::lower_bound(entries.begin(), entries.end(), std::pair{dest, 0u});
  auto end = std::upper_bound(begin, entries.end(), std::pair{dest, UINT32_MAX});
  if (end - begin <= 1)
    return;
  // An explicit no-jump hook on a wrapper only receives callers of that wrapper.
  bool wrapper = address != module->addr(module->seg_off[0] + dest);
  auto offset = read<int32_t>(db->methods, 12ull * method + 8);
  std::unordered_map<uint32_t, Layout> owners;
  bool distinguishable = true, compatible = true;
  for (auto it = begin; it != end; ++it) {
    auto k = it->second;
    auto row = 12ull * k;
    if (wrapper && read<int32_t>(db->methods, row + 8) != offset)
      continue;
    auto candidate = k == method ? layout : describe(k);
    if (candidate.stack_bytes != layout.stack_bytes)
      throw std::runtime_error("Shared entry has different stack argument sizes");
    auto owner = read<uint32_t>(db->methods, row) & id_mask;
    bool instance = !(read<uint16_t>(db->method_impl(k), 4) & 0x10);
    distinguishable &= instance && !derives(owner, db->find("System.ValueType"));
    compatible &= layout == candidate;
    auto [where, fresh] = owners.emplace(owner, candidate);
    if (!fresh && !(where->second == candidate))
      throw std::runtime_error("Same receiver has incompatible signatures at shared entry");
  }
  if (!distinguishable && !compatible)
    throw std::runtime_error("Shared entry has no receiver to distinguish incompatible signatures");
  if (distinguishable && owners.size() > 1) {
    for (auto &[owner, candidate] : owners)
      site.owners.push_back(owner);
    std::sort(site.owners.begin(), site.owners.end());
  }
  log_message(ANDROID_LOG_INFO, site.owners.empty()
      ? "Shared hook entry: methods with identical calling convention share callbacks"
      : "Shared hook entry: dispatching by runtime receiver type");
}
uintptr_t class_link(uintptr_t address) {
  if (!address || address % alignof(uintptr_t))
    return 0;
  uintptr_t link{};
  iovec src{reinterpret_cast<void *>(address), sizeof(link)}, dst{&link, sizeof(link)};
  return process_vm_readv(getpid(), &dst, 1, &src, 1, 0) == sizeof(link) ? link : 0;
}
uint32_t class_row(uintptr_t type) {
  auto offset = type - reinterpret_cast<uintptr_t>(db->types.data());
  return offset < db->types.size() && offset % 72 == 0 ? offset / 72 : 0;
}
uint32_t receiver_owner(const Site &site, uintptr_t object) {
  for (auto id = class_row(class_link(class_link(object))); id; id = db->parent(id))
    if (std::binary_search(site.owners.begin(), site.owners.end(), id))
      return id;
  return 0;
}
} // namespace
void initialize_native_hooks(lua_State *L, Tdb &t, const Module &m) {
  module = &m;
  db = &t;
  vm = L;
  ++epoch;
}
bool native_hooks_in_flight() { return native_depth != 0; }
uint64_t register_native_hook(lua_State *L, uint32_t method, int pre, int post,
                              bool ignore_jump) {
  auto layout = describe(method);
  auto address = native_method_address(L, method);
  index_entries(address - read<int32_t>(db->methods, 12ull * method + 8));
  auto dest = destinations.at(method);
  if (dest == UINT32_MAX)
    throw std::runtime_error("Method has no callable entry in the initialized TDB");
  if (!ignore_jump)
    address = module->addr(module->seg_off[0] + dest);
  Site *site;
  if (auto i = by_address.find(address); i != by_address.end()) {
    site = i->second;
  } else {
    auto value = std::make_unique<Site>();
    describe_site(*value, method, layout, address);
    site = value.get();
    // Keep all patch records for the process lifetime, including across Lua
    // reloads.
    sites.push_back(std::move(value));
    install_native_entry(*module, address, &site->patch);
    by_address.emplace(address, site);
  }
  auto s = std::make_shared<Subscriber>();
  s->serial = ++serial;
  s->epoch = epoch;
  s->method = method;
  s->owner = read<uint32_t>(db->methods, 12ull * method) & id_mask;
  s->layout = std::move(layout);
  s->pre = pre;
  s->post = post;
  site->subscribers.push_back(s);
  subscribers.emplace(s->serial, s);
  site->active.store(site->subscribers.size(), std::memory_order_release);
  auto owner = read<uint32_t>(db->methods, 12ull * method) & id_mask;
  log_message(ANDROID_LOG_INFO,
              ("Native hook installed: " + db->full_name(owner) + "." +
               std::string(db->method(method, owner)))
                  .c_str());
  return s->serial;
}
void remove_native_hook(uint64_t id) {
  auto it = subscribers.find(id);
  if (it == subscribers.end())
    return;
  it->second->enabled = false;
  subscribers.erase(it);
  // Registry references belong to lua_mod_hooks, which releases them.
  for (auto &site : sites) {
    std::erase_if(site->subscribers,
                  [&](const auto &s) { return s->serial == id; });
    site->active.store(site->subscribers.size(), std::memory_order_release);
  }
}
void dispatch_native(NativeFrame &frame) {
  auto &site = *static_cast<Site *>(frame.site);
  if (site.active.load(std::memory_order_acquire) == 0) {
    frame.original = site.patch.original;
    frame.stack_size = site.stack_bytes;
    frame.stack_copy = reinterpret_cast<const uint8_t *>(frame.caller_sp);
    rise_native_call(&frame);
    return;
  }
  Invocation invocation;
  frame.original = site.patch.original;
  frame.stack_size = site.stack_bytes;
  invocation.stack.resize(frame.stack_size);
  if (frame.stack_size)
    memcpy(invocation.stack.data(),
           reinterpret_cast<const void *>(frame.caller_sp), frame.stack_size);
  frame.stack_copy = invocation.stack.data();
  bool skip = false;
  {
    std::lock_guard lock(lua_mutex());
    invocation.generation = epoch;
    uint32_t owner = site.owners.empty() ? 0 : receiver_owner(site, frame.x[1]);
    for (auto &s : site.subscribers)
      if (s->enabled && s->epoch == epoch && (site.owners.empty() || owner == s->owner))
        invocation.callbacks.push_back(s);
    invocation.storage.resize(invocation.callbacks.size(), LUA_NOREF);
    if (!invocation.callbacks.empty()) {
      ++native_depth;
      ActiveCall active(invocation);
      invocation.top = lua_gettop(vm);
      try {
        build_arguments(invocation, frame);
        skip = before(invocation, frame);
      } catch (const std::exception &e) {
        log_message(ANDROID_LOG_ERROR, e.what());
      }
      lua_settop(vm, invocation.top);
      --native_depth;
    }
  }
  if (!skip)
    rise_native_call(&frame);
  else {
    memset(frame.result_x, 0, sizeof(frame.result_x));
    memset(frame.result_v, 0, sizeof(frame.result_v));
    if (invocation.layout().result.kind == Kind::Value && frame.x[8])
      memset(reinterpret_cast<void *>(frame.x[8]), 0, invocation.layout().result.size);
  }
  {
    std::lock_guard lock(lua_mutex());
    if (invocation.generation == epoch && !invocation.callbacks.empty()) {
      ++native_depth;
      ActiveCall active(invocation);
      int top = lua_gettop(vm);
      after(invocation, frame);
      lua_settop(vm, top);
      --native_depth;
    }
  }
}
int hook_native_field(lua_State *L) {
  const char *name = luaL_checkstring(L, 3);
  luaL_checktype(L, 2, LUA_TTABLE);
  bool write = lua_toboolean(L, lua_upvalueindex(1));
  try {
    auto p = bits(L, 1);
    for (auto inv = current; inv; inv = inv->parent) {
      for (const auto &[address, t] : inv->buffers) {
        if (address != p)
          continue;
        push_type_definition(L, *db, t.id);
        bool matches = lua_rawequal(L, -1, 2);
        lua_pop(L, 1);
        if (!matches)
          continue;
        auto declared = t.kind == Kind::ByRef ? type(t.id) : t;
        if (declared.kind != Kind::Value)
          throw std::runtime_error("Expected a structured value buffer");
        return access_value_field(L, *db, t.id,
            {reinterpret_cast<uint8_t *>(p), declared.size}, name, write);
      }
    }
    throw std::runtime_error("Expected a matching value buffer in the active hook");
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int is_managed_argument(lua_State *L) {
  if (is_live_object(L, 1)) {
    lua_pushboolean(L, true);
    return 1;
  }
  uintptr_t object = 0;
  if (lua_islightuserdata(L, 1))
    object = reinterpret_cast<uintptr_t>(lua_touserdata(L, 1));
  else if (lua_isinteger(L, 1))
    object = static_cast<uintptr_t>(lua_tointeger(L, 1));
  auto info = class_link(object);
  auto type = class_link(info);
  auto id = class_row(type);
  bool valid = false;
  if (id) {
    auto row = size_t(id) * 72;
    auto canonical = read<uintptr_t>(db->types, row + 64);
    // Per-instance virtual tables can have a different object-info pointer.
    valid = read<uintptr_t>(db->types, row + 56) &&
            (canonical == info || class_link(canonical) == type);
  }
  lua_pushboolean(L, valid);
  return 1;
}
int hook_to_managed(lua_State *L) {
  try {
    if (is_live_object(L, 1)) {
      lua_settop(L, 1);
      return 1;
    }
    auto address = bits(L, 1);
    if (!address) {
      lua_pushnil(L);
      return 1;
    }
    if (push_cached_live_object(L, address))
      return 1;
    auto *invocation = current;
    while (invocation) {
      if (invocation->objects.contains(address)) {
        push_live_object(L, address, 0);
        return 1;
      }
      invocation = invocation->parent;
    }
    throw std::runtime_error(
        "Pointer is not a managed argument of the active hook");
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int hook_to_ptr(lua_State *L) {
  try {
    auto value = bits(L, 1);
    if (current && is_live_object(L, 1)) {
      current->objects.emplace(value, 0);
      lua_pushvalue(L, 1);
      current->held_objects.push_back(luaL_ref(L, LUA_REGISTRYINDEX));
    }
    push_bits(L, value);
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int hook_to_integer(lua_State *L) {
  try {
    lua_pushinteger(L, bits(L, 1));
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int hook_to_number(lua_State *L) {
  bool wide = lua_toboolean(L, lua_upvalueindex(1));
  if (lua_isnumber(L, 1)) {
    lua_settop(L, 1);
    return 1;
  }
  try {
    auto raw = bits(L, 1);
    double value;
    if (wide)
      memcpy(&value, &raw, 8);
    else {
      float f;
      memcpy(&f, &raw, 4);
      value = f;
    }
    lua_pushnumber(L, value);
    return 1;
  } catch (const std::exception &e) {
    lua_pushstring(L, e.what());
  }
  return lua_error(L);
}
int hook_number_to_ptr(lua_State *L) {
  bool wide = lua_toboolean(L, lua_upvalueindex(1));
  double number = luaL_checknumber(L, 1);
  uint64_t raw{};
  if (wide)
    memcpy(&raw, &number, 8);
  else {
    float f = number;
    memcpy(&raw, &f, 4);
  }
  push_bits(L, raw);
  return 1;
}
} // namespace mod_lua
extern "C" void rise_native_dispatch(mod_lua::NativeFrame *frame) {
  mod_lua::dispatch_native(*frame);
}
