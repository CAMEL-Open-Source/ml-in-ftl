/**
 * predictor_impl.hh
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
#include "ftl/predictor_interface.hh"

namespace SimpleSSD::ML {

class FileModel : public AbstractMLModel {
 protected:
  std::unordered_map<uint64_t, CoReadPrediction> predictTable;

  uint64_t totalLogicalPages;
  CoReadPrediction readEntry(std::ifstream &predictionFile);

 public:
  FileModel(ObjectData &);
  ~FileModel();

  void init(CoReadPredictor *, uint64_t totalLogicalPages);
  void inferenceDone(uint64_t, uint64_t) override;
  void trainingDone(uint64_t, uint64_t) override;
};

}