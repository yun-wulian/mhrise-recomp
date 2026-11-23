#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nvn {

// Recorded commands are trivially copyable and can outlive both the resource
// and its guest address. Slots have stable addresses and are reused by generation.
template<class T> class ResourceSlots;
template<class T> class ResourceRef {
    friend class ResourceSlots<T>;
    struct Slot {
        std::atomic<uint64_t> generation{1};
        std::atomic<T*> value{nullptr};
    };
    union { T* direct_; Slot* slot_; };
    uint64_t generation_ = 0;
    ResourceRef(Slot* slot, uint64_t generation) : slot_(slot), generation_(generation) {}
public:
    // Executor-owned images and fixture resources have a directly scoped lifetime.
    ResourceRef(T* direct = nullptr) : direct_(direct) {}
    T* get() const {
        if (!generation_) return direct_;
        T* value = slot_->value.load(std::memory_order_acquire);
        return slot_->generation.load(std::memory_order_acquire) == generation_ ? value : nullptr;
    }
    bool assigned() const { return generation_ || direct_; }
    operator T*() const { return get(); }
    T* operator->() const { return get(); }
};

// Allocation and retirement are serialized by the owning registry. Rendering
// resolves references under the executor lock, which also serializes destruction.
template<class T> class ResourceSlots {
    using Slot = typename ResourceRef<T>::Slot;
    std::deque<Slot> slots_;
    std::vector<Slot*> free_;
    inline static std::atomic<uint64_t> retirements_{0};
public:
    // Descriptor snapshots retain references whose slot can be invalidated independently of the pool.
    static uint64_t retirements() { return retirements_.load(std::memory_order_acquire); }
    ResourceRef<T> insert(T* value) {
        Slot* slot;
        if (free_.empty()) slot = &slots_.emplace_back();
        else { slot = free_.back(); free_.pop_back(); }
        slot->value.store(value, std::memory_order_release);
        return {slot, slot->generation.load(std::memory_order_relaxed)};
    }
    void erase(ResourceRef<T> ref) {
        Slot& slot = *ref.slot_;
        slot.generation.fetch_add(1, std::memory_order_release);
        slot.value.store(nullptr, std::memory_order_release);
        retirements_.fetch_add(1, std::memory_order_release);
        free_.push_back(&slot);
    }
};

template<class T> class ResourceRegistry {
    mutable std::mutex mutex_;
    ResourceSlots<T> slots_;
    struct Entry { std::shared_ptr<T> value; ResourceRef<T> ref; };
    std::unordered_map<const void*, Entry> entries_;
public:
    std::shared_ptr<T> acquire(const void* key) const {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(key);
        return it == entries_.end() ? nullptr : it->second.value;
    }
    ResourceRef<T> reference(const void* key) const {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(key);
        return it == entries_.end() ? ResourceRef<T>{} : it->second.ref;
    }
    T* get(const void* key) const { return reference(key).get(); }
    T& create(const void* key) {
        std::lock_guard lock(mutex_);
        auto [it, inserted] = entries_.try_emplace(key);
        auto& entry = it->second;
        if (!inserted) slots_.erase(entry.ref);
        entry.value = std::make_shared<T>();
        entry.ref = slots_.insert(entry.value.get());
        return *entry.value;
    }
    std::shared_ptr<T> take(const void* key) {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) return {};
        slots_.erase(it->second.ref);
        auto value = std::move(it->second.value);
        entries_.erase(it);
        return value;
    }
};
}
