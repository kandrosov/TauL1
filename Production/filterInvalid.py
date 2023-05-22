import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()


cpp_str = """
static std::map<int, bool> batch_validity;

void fill_batch_validity(ROOT::RDF::RNode df, int batch_size)
{
  df.Foreach([&](bool is_valid, ULong64_t rdfentry) {
    const int batch_id = rdfentry / batch_size;
    if(batch_validity.count(batch_id)) {
      batch_validity[batch_id] = batch_validity[batch_id] && is_valid;
    } else {
      batch_validity[batch_id] = is_valid;
    }
  }, {"is_valid", "rdfentry_"});
}

bool batch_is_valid(ULong64_t rdfentry, int batch_size)
{
  const int batch_id = rdfentry / batch_size;
  auto iter = batch_validity.find(batch_id);
  if(iter == batch_validity.end())
    throw std::runtime_error("batch_is_valid: batch_id not found");
  return iter->second;
}
"""

if not ROOT.gInterpreter.Declare(cpp_str):
  raise RuntimeError(f'Failed to declare cpp code')

def get_columns(df):
  all_columns = [ str(c) for c in df.GetColumnNames() ]
  simple_types = [ 'Int_t', 'UInt_t', 'Long64_t', 'ULong64_t', 'int', 'long' ]
  column_types = { c : str(df.GetColumnType(c)) for c in all_columns }
  all_columns = sorted(all_columns, key=lambda c: (column_types[c] not in simple_types, c))
  return all_columns, column_types

def ListToVector(list, type="string"):
	vec = ROOT.std.vector(type)()
	for item in list:
		vec.push_back(item)
	return vec

def filter_invalid(input, output, tree_name, batch_size):
  df = ROOT.RDataFrame(tree_name, input)
  columns, column_types = get_columns(df)
  ROOT.fill_batch_validity(ROOT.RDF.AsRNode(df), batch_size)
  df = df.Define('is_batch_valid', f'batch_is_valid(rdfentry_, {batch_size})')
  df = df.Filter('is_batch_valid')
  opt = ROOT.RDF.RSnapshotOptions()
  opt.fCompressionAlgorithm = ROOT.ROOT.kLZ4
  opt.fCompressionLevel = 5
  opt.fMode = 'RECREATE'
  columns_v = ListToVector(columns)
  df.Snapshot(tree_name, output, columns_v, opt)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Filter invalid entries.')
  parser.add_argument('--input', required=True, type=str, help="input file")
  parser.add_argument('--output', required=True, type=str, help="output file")
  parser.add_argument('--tree', required=True, type=str, help="tree name")
  parser.add_argument('--batch-size', required=True, type=int, help="batch size")
  args = parser.parse_args()

  filter_invalid(args.input, args.output, args.tree, args.batch_size)
