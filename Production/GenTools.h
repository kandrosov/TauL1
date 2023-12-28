#pragma once

#include "AnalysisTools.h"
#include "GenLepton.h"

inline bool IsLepton(TauType tau_type)
{
  static const std::set<TauType> lepton_types = {
    TauType::e, TauType::mu, TauType::tau, TauType::emb_e, TauType::emb_mu, TauType::emb_tau
  };
  return lepton_types.count(tau_type);
}

inline RVecLV GetGenP4(const RVecI& tauType, const RVecI& genLepUniqueIdx, const RVecI& genJetUniqueIdx,
                       const ROOT::VecOps::RVec<reco_tau::gen_truth::GenLepton>& genLeptons,
                       const RVecLV& genJet_p4)
{
  RVecLV gen_p4(tauType.size());
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(IsLepton(tau_type)) {
      const int genLepton_idx = genLepUniqueIdx[tau_idx];
      gen_p4[tau_idx] = genLeptons.at(genLepton_idx).visibleP4();
    } else if(tau_type == TauType::jet) {
      const int genJet_idx = genJetUniqueIdx[tau_idx];
      gen_p4[tau_idx] = genJet_p4.at(genJet_idx);
    }
  }
  return gen_p4;
}

inline RVecI GetGenCharge(const RVecI& tauType, const RVecI& genLepUniqueIdx,
                          const ROOT::VecOps::RVec<reco_tau::gen_truth::GenLepton>& genLeptons)
{
  RVecI gen_charge(tauType.size(), 0);
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(IsLepton(tau_type)) {
      const int genLepton_idx = genLepUniqueIdx[tau_idx];
      gen_charge[tau_idx] = genLeptons.at(genLepton_idx).charge();
    }
  }
  return gen_charge;
}

inline RVecI GetGenPartonFlavour(const RVecI& tauType, const RVecI& genJetUniqueIdx,
                                 const ROOT::VecOps::RVec<short>& GenJet_partonFlavour)
{
  RVecI gen_pf(tauType.size(), -999);
  for(size_t tau_idx = 0; tau_idx < tauType.size(); ++tau_idx) {
    const auto tau_type = static_cast<TauType>(tauType[tau_idx]);
    if(tau_type == TauType::jet) {
      const int genJet_idx = genJetUniqueIdx[tau_idx];
      gen_pf[tau_idx] = GenJet_partonFlavour.at(genJet_idx);
    }
  }
  return gen_pf;
}

inline RVecI GetTauTypes(const ROOT::VecOps::RVec<reco_tau::gen_truth::GenLepton>& genLeptons,
                         const RVecI& genLepUniqueIdx, const RVecSetInt& genLepIndices,
                         const RVecI& genJetUniqueIdx, bool isEmbedded)
{
  using namespace reco_tau::gen_truth;
  RVecI tau_types(genLepUniqueIdx.size(), static_cast<int>(TauType::other));
  for(size_t obj_idx = 0; obj_idx < tau_types.size(); ++obj_idx) {
    if(genLepIndices[obj_idx].empty()) {
      if(genJetUniqueIdx[obj_idx] >= 0)
        tau_types[obj_idx] = static_cast<int>(isEmbedded ? TauType::emb_jet : TauType::jet);
    } else {
      if(genLepUniqueIdx[obj_idx] >= 0 && genLepIndices[obj_idx].size() == 1) {
        const int genLepton_idx = genLepUniqueIdx[obj_idx];
        const auto& genLepton = genLeptons[genLepton_idx];
        if(genLepton.kind() == GenLepton::Kind::TauDecayedToHadrons)
          tau_types[obj_idx] = static_cast<int>(isEmbedded ? TauType::emb_tau : TauType::tau);
        else if(genLepton.kind() == GenLepton::Kind::PromptElectron
            || genLepton.kind() == GenLepton::Kind::TauDecayedToElectron)
          tau_types[obj_idx] = static_cast<int>(isEmbedded ? TauType::emb_e : TauType::e);
        else if(genLepton.kind() == GenLepton::Kind::PromptMuon
            || genLepton.kind() == GenLepton::Kind::TauDecayedToMuon)
          tau_types[obj_idx] = static_cast<int>(isEmbedded ? TauType::emb_mu : TauType::mu);
      }
    }
  }
  return tau_types;
}

inline RVecVecI GetL1TauTowerVar(size_t n_l1Taus, const RVecI& l1TauTowerVar, const RVecI& l1TauTower_l1TauIdx)
{
  RVecVecI converted;
  converted.resize(n_l1Taus);
  if(l1TauTowerVar.size() != l1TauTower_l1TauIdx.size()) {
    std::cout << "l1TauTowerVar and l1TauTower_l1TauIdx have different sizes "
              << l1TauTowerVar.size() << " != " << l1TauTower_l1TauIdx.size() << std::endl;
    throw std::runtime_error("l1TauTowerVar and l1TauTower_l1TauIdx have different sizes");
  }
  for(size_t tower_idx = 0; tower_idx < l1TauTowerVar.size(); ++tower_idx) {
    const int l1Tau_idx = l1TauTower_l1TauIdx[tower_idx];
    if(l1Tau_idx < 0 || static_cast<size_t>(l1Tau_idx) >= n_l1Taus) {
      std::cout << "l1TauTower_l1TauIdx has invalid value " << l1Tau_idx << std::endl;
      throw std::runtime_error("l1TauTower_l1TauIdx has invalid value");
    }
    converted.at(l1Tau_idx).push_back(l1TauTowerVar[tower_idx]);
  }
  return converted;
}

inline int GetFirstCopy(int idx, const RVecI& GenPart_pdgId, const RVecI& GenPart_genPartIdxMother)
{
  while(GenPart_genPartIdxMother[idx] >= 0 && static_cast<size_t>(GenPart_genPartIdxMother[idx]) < GenPart_pdgId.size()
      && GenPart_pdgId[idx] == GenPart_pdgId[GenPart_genPartIdxMother[idx]])
    idx = GenPart_genPartIdxMother[idx];
  if(idx < 0 || static_cast<size_t>(idx) >= GenPart_genPartIdxMother.size())
    idx = -1;
  return idx;
}

inline std::set<int> FindDaughters(int idx, const RVecI& GenPart_pdgId, const RVecI& GenPart_genPartIdxMother)
{
  std::set<int> daughters;
  for(size_t p_idx = static_cast<size_t>(idx) + 1; p_idx < GenPart_pdgId.size(); ++p_idx) {
    if(GenPart_genPartIdxMother[p_idx] == idx)
      daughters.insert(p_idx);
  }
  return daughters;

}

inline std::set<int> FindParticles(int pdgId, const std::vector<int>& expected_daughters, const RVecI& GenPart_pdgId,
                                   const RVecI& GenPart_genPartIdxMother)
{
  std::set<int> selected_particles;
  for(size_t p_idx = 0; p_idx < GenPart_pdgId.size(); ++p_idx) {
    if(GenPart_pdgId[p_idx] != pdgId) continue;
    const auto daughter_idx = FindDaughters(p_idx, GenPart_pdgId, GenPart_genPartIdxMother);
    std::set<int> selected_daughters;
    bool all_matched = true;
    for(auto expected_daughter_pdg : expected_daughters) {
      bool matched = false;
      for(auto daughter_idx : daughter_idx) {
        if(!selected_daughters.count(daughter_idx) && GenPart_pdgId[daughter_idx] == expected_daughter_pdg) {
          selected_daughters.insert(daughter_idx);
          matched = true;
          break;
        }
      }
      if(!matched) {
        all_matched = false;
        break;
      }
    }
    if(all_matched)
      selected_particles.insert(p_idx);
  }
  return selected_particles;
}