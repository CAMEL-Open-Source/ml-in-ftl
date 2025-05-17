/**
 * training_impl.hh
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

#include <list>
#include <map>
#include <set>

#include "ftl/predictor_interface.hh"
#include "sim/object.hh"
#include "sim/types.hh"

namespace torch {
namespace optim {
class Optimizer;
}
}  // namespace torch

namespace SimpleSSD::ML {

class AbstractTrainig : public Object {
 protected:
  uint64_t totalLogicalPages;
  CoReadPredictor *pp;

 public:
  AbstractTrainig(ObjectData &o) : Object(o) {}
  virtual ~AbstractTrainig() = default;

  virtual void init(CoReadPredictor *pp, uint64_t totalLogicalPages) {
    this->pp = pp;
    this->totalLogicalPages = totalLogicalPages;
  }
  virtual void add(WindowEntry &) = 0;
  virtual bool get(LPN, CoReadPrediction &) = 0;

  void getStatList(std::vector<Stat> &, std::string) noexcept override {}
  void getStatValues(std::vector<double> &) noexcept override {}
  void resetStatValues() noexcept override {}

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

class MLPModel;

class MLPTrainig : public AbstractTrainig {
 private:
  MLPModel *model;
  ::torch::optim::Optimizer *optimizer;

 public:
  MLPTrainig(ObjectData &o);
  ~MLPTrainig();

  void add(WindowEntry &) override;
  bool get(LPN, CoReadPrediction &) override;

  void getStatList(std::vector<Stat> &, std::string) noexcept override {}
  void getStatValues(std::vector<double> &) noexcept override {}
  void resetStatValues() noexcept override {}

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

}  // namespace SimpleSSD::ML
