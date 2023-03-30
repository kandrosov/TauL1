import os
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

def ListToVector(list, type="string"):
	vec = ROOT.std.vector(type)()
	for item in list:
		vec.push_back(item)
	return vec

headers_dir = os.path.dirname(os.path.abspath(__file__))
for header in [ 'AnalysisTools.h', 'GenLepton.h', 'TupleMaker.h' ]:
  header_path = os.path.join(headers_dir, header)
  if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
    raise RuntimeError(f'Failed to load {header_path}')

sample_path = "/eos/cms/store/group/phys_tau/kandroso/Run3_HLT/prod_v1/GluGluHToTauTau_M-125"
#files = '*.root'
files = 'nano_2.root'
df_in = ROOT.RDataFrame("Events", os.path.join(sample_path, files))

df_in = df_in.Define("genLeptons", """
  reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass,
                                              GenPart_genPartIdxMother, GenPart_pdgId, GenPart_statusFlags, event)
""")
df_in = df_in.Define('L1Tau_mass', 'RVecF(L1Tau_pt.size(), 0.)')
df_in = df_in.Define('L1Tau_p4', 'GetP4(L1Tau_pt, L1Tau_eta, L1Tau_phi, L1Tau_mass)')
df_in = df_in.Define('GenLepton_p4', 'v_ops::visibleP4(genLeptons)')

df_in = df_in.Define('L1Tau_genLepIndices', 'FindMatchingSet(L1Tau_p4, GenLepton_p4, 0.4)')
df_in = df_in.Define('L1Tau_genLepUniqueIdx', 'FindUniqueMatching(L1Tau_p4, GenLepton_p4, 0.4)')
df_in = df_in.Define('L1Tau_sel', '''
  RVecB indices(L1Tau_pt.size(), false);
  for(size_t l1Tau_idx = 0; l1Tau_idx < indices.size(); ++l1Tau_idx) {
    if(L1Tau_genLepUniqueIdx[l1Tau_idx] >= 0 && L1Tau_genLepIndices[l1Tau_idx].size() == 1) {
      const int genLepton_idx = L1Tau_genLepUniqueIdx[l1Tau_idx];
      const auto& genLepton = genLeptons[genLepton_idx];
      if(genLepton.kind() == reco_tau::gen_truth::GenLepton::Kind::TauDecayedToHadrons
          && genLepton.visibleP4().pt() > 20 && std::abs(genLepton.visibleP4().eta()) < 2.1) {
        indices[l1Tau_idx] = true;
      }
    }
  }
  return indices;
''')

df_in = df_in.Define('nSel', 'L1Tau_pt[L1Tau_sel].size()')
nTau = int(df_in.Sum('nSel').GetValue())
print(f'Number of input taus = {nTau}')
df_out = ROOT.RDataFrame(nTau)

df_in = df_in.Define('L1Tau_pt_sel', 'L1Tau_pt[L1Tau_sel]')
df_in = df_in.Define('L1Tau_eta_sel', 'L1Tau_eta[L1Tau_sel]')
df_in = df_in.Define('L1Tau_phi_sel', 'L1Tau_phi[L1Tau_sel]')

columns_in = [ 'L1Tau_pt_sel', 'L1Tau_eta_sel', 'L1Tau_phi_sel']
columns_out = [ 'L1Tau_pt', 'L1Tau_eta', 'L1Tau_phi' ]
columns_in_v = ListToVector(columns_in)
columns_out_v = ListToVector(columns_out)

column_types = [ str(df_in.GetColumnType(c)) for c in columns_in ]
tuple_maker = ROOT.analysis.TupleMaker(*column_types)(10, nTau)
df_in_node = ROOT.RDF.AsRNode(df_in)
df_out_node = ROOT.RDF.AsRNode(df_out)
df_out = tuple_maker.process(df_in_node, df_out_node, columns_in_v, 0, 100, 100)
for column_idx in range(len(columns_in)):
  df_out = df_out.Define(columns_out[column_idx], f'_entry.valid ? _entry.float_values.at({column_idx}): 0.f')
#print("Creating snapshot...")

opt = ROOT.RDF.RSnapshotOptions()
opt.fCompressionAlgorithm = ROOT.ROOT.kLZMA
opt.fCompressionLevel = 9
opt.fMode = 'RECREATE'
df_out.Snapshot("Events", "test.root", columns_out_v, opt)
#print("Snapshot created.")
tuple_maker.join()
print("L1 tuple has been created.")
