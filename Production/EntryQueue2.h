/*! Definiton of a thread-safe fixed size entry queue. */

#pragma once

#include <mutex>
#include <queue>
#include <condition_variable>

namespace analysis {

template<typename Entry>
class EntryQueue {
public:
  using Queue = std::queue<Entry>;
  using Mutex = std::mutex;
  using Lock = std::unique_lock<Mutex>;
  using CondVar = std::condition_variable;

public:
  explicit EntryQueue(size_t max_size, size_t max_entries = std::numeric_limits<size_t>::max())
    : max_size_(max_size), max_entries_(max_entries), n_entries_(0), all_done_(false)
  {
  }

  bool Push(const Entry& entry)
  {
    {
      Lock lock(mutex_);
      if(n_entries_ >= max_entries_)
        return false;
      cond_var_.wait(lock, [&] { return queue_.size() < max_size_; });
      //std::cout << "nEntries is "<<n_entries_ << std::endl;
      queue_.push(entry);
      ++n_entries_;
    }
    cond_var_.notify_all();
    return true;
  }

  bool Pop(Entry& entry)
  {
    bool entry_is_valid = false;;
    {
      Lock lock(mutex_);
      cond_var_.wait(lock, [&] { return queue_.size() || all_done_; });
      if(!queue_.empty()) {
        entry = queue_.front();
        entry_is_valid = true;
        queue_.pop();
      }
    }
    cond_var_.notify_all();
    return entry_is_valid;
  }

  void SetAllDone(bool value = true)
  {
    {
      Lock lock(mutex_);
      all_done_ = value;
      //std::cout << "all done set "<< std::endl;
    }
    cond_var_.notify_all();
  }

private:
  Queue queue_;
  const size_t max_size_, max_entries_;
  size_t n_entries_;
  bool all_done_;
  Mutex mutex_;
  CondVar cond_var_;
};

} // namespace analysis


template<typename T>
void addEntry(Entry& entry, int index, const T& value)
{
  entry.Add(index, value);
}

template<typename ...Args>
void fillEntry(Entry& entry, Args&& ...args)
{
    int index = 0;
    std::initializer_list<int>{ (addEntry(entry, index++, args), 0)... };
}


template<typename T>
void addEntry(std::shared_ptr<Entry>& entry, int index, ) {}

template<typename T,typename ...Args>
void putEntry(std::shared_ptr<Entry>& entry, int var_index,
              const T& value, Args&& ...args)
{
  //std::cout << var_index << "\t " << value <<std::endl;
  entry->Add(var_index, value);
  //std::cout << "before incrementing " << var_index << std::endl;
  //std::cout << "after incrementing " << var_index << std::endl;
  putEntry(entry, var_index+1,std::forward<Args>(args)...);
}