/**
 * workload_monitor.cc
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

#include "ftl/ml-prediction/interface/workload_monitor.hh"

#include "ftl/ml-prediction/interface/predictor.hh"
#include "ftl/ml-prediction/interface/predictor_impl.hh"
#include "ftl/predictor_interface.hh"

namespace SimpleSSD::ML {

WorkloadMonitor::WorkloadMonitor(ObjectData &o)
    : Object(o),
      pPredictor(nullptr),
      lastRequestTime(0),
      lastRead(UINT64_MAX) {}

WorkloadMonitor::~WorkloadMonitor() {
  warn_if(!window.empty(), "WorkloadMonitor | window is not empty %lu",
          window.size());
}

void WorkloadMonitor::init(CoReadPredictor *pp) {
  pPredictor = pp;
}

void WorkloadMonitor::enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn,
                                        uint32_t nlp) {
  uint64_t now = getTick();

  UNUSED(isRead);

  auto ret = window.emplace(tag, WindowEntry(slpn, nlp, now - lastRequestTime));

  // Note that enqueueNewRequest can be called on filling phase
  lastRequestTime = now;
  lastRead = tag;

  if (!ret.second) {
    abort();
  }

  lastRead = tag;
}

void WorkloadMonitor::dequeueNewRequest(uint64_t tag) {
  auto iter = window.find(tag);
  if (iter == window.end()) {
    return;
  }

  pPredictor->pModel->train(std::move(iter->second));
  window.erase(iter);
}

void WorkloadMonitor::getStatList(std::vector<Stat> &list,
                                  std::string prefix) noexcept {
  UNUSED(list);
  UNUSED(prefix);
}

void WorkloadMonitor::getStatValues(std::vector<double> &values) noexcept {
  UNUSED(values);
}

void WorkloadMonitor::resetStatValues() noexcept {}

}  // namespace SimpleSSD::ML
