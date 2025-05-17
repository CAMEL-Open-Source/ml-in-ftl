/**
 * workload_monitor.hh
 * Copyright (C) 2021-2024 CAMELab
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#ifndef __SIM_CO_READ_WORKLOAD_MONITOR_HH__
#define __SIM_CO_READ_WORKLOAD_MONITOR_HH__

#include <deque>

#include "ftl/predictor_interface.hh"
#include "sim/object.hh"
namespace SimpleSSD::ML {

class CoReadPredictor;

// monitor average read size
// monitor co-read window
class WorkloadMonitor : public Object {
 private:
  CoReadPredictor *pPredictor;

  std::unordered_map<uint64_t, WindowEntry> window;
  uint64_t lastRequestTime;
  uint64_t lastRead;

 public:
  WorkloadMonitor(ObjectData &o);
  ~WorkloadMonitor();

  void init(CoReadPredictor *);

  void enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn, uint32_t nlp);
  void dequeueNewRequest(uint64_t tag);

  void getStatList(std::vector<Stat> &, std::string) noexcept;
  void getStatValues(std::vector<double> &) noexcept;
  void resetStatValues() noexcept;

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

}  // namespace SimpleSSD::ML

#endif