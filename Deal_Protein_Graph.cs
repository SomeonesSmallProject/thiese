namespace Protein_Graph
{
    public class Protein
    {
        private Dictionary<int, string> AlphaCarben = new Dictionary<int, string>();
        private Dictionary<int, int[]> Helix = new Dictionary<int, int[]>();
        private Dictionary<int, int> HelixType = new Dictionary<int, int>();
        private List<string[]> SheetList = new List<string[]>();
        private Dictionary<int, List<double>> CarbenPosition = new Dictionary<int, List<double>>();
        public static String ListCharToString(List<char> chars)
        {
            string result = "";
            for (int i = 0; i < chars.Count(); i = i + 1)
            {
                result = result + chars[i].ToString();
            }
            return (result);
        }
        public void GetAAName(string AtomString)
        {
            Dictionary<string, string> AA_Dict = new Dictionary<string, string>{
    {"ALA","alanine"},
    {"ARG","arginine"},
    {"ASN","asparagine"},
    {"ASP","aspartic acid"},
    {"CYS","cysteine"},
    {"GLN","glutamine"},
    {"GLU","glutamic acid"},
    {"GLY","glycine"},
    {"HIS","histidine"},
    {"ILE","isoleucine"},
    {"LEU","leucine"},
    {"LYS","lysine"},
    {"MET","methionine"},
    {"PHE","phenylalanine"},
    {"PRO","proline"},
    {"SER","serine"},
    {"THR","threonine"},
    {"TRP","tryptophan"},
    {"TYR","tyrosine"},
    {"VAL","valine"}
            };
            List<Char> tmp_string_list = AtomString.ToList();
            string aa = ListCharToString(tmp_string_list.Skip(17).Take(3).ToList()).ToUpper();
            int label = int.Parse(ListCharToString(tmp_string_list.Skip(23).Take(3).ToList()));
            string AA_Name = AA_Dict[aa];
            AlphaCarben.Add(label, AA_Name);
        }
        public void GetPosition(string AtomString)
        {
            List<char> tmp_string_list = AtomString.ToList();
            string X = ListCharToString(tmp_string_list.Skip(30).Take(8).ToList());
            string Y = ListCharToString(tmp_string_list.Skip(38).Take(8).ToList());
            string Z = ListCharToString(tmp_string_list.Skip(46).Take(8).ToList());
            string Label = ListCharToString(tmp_string_list.Skip(23).Take(3).ToList());
            List<double> Position = new List<double> { double.Parse(X), double.Parse(Y), double.Parse(Z) };
            CarbenPosition!.Add(int.Parse(Label), Position);
        }
        public void DealAtoms(string AtomString)
        {
            List<char> tmp_string_list = AtomString.ToList();
            if (ListCharToString(tmp_string_list.Skip(13).Take(2).ToList()).Equals("CA"))
            {
                GetAAName(AtomString);
                GetPosition(AtomString);
            }
        }
        public void DealHelix(string HelixString)
        {
            List<char> tmp_string_list = HelixString.ToList();
            int HelixNumber = int.Parse(ListCharToString(tmp_string_list.Skip(7).Take(3).ToList()));
            int StartHelix = int.Parse(ListCharToString(tmp_string_list.Skip(21).Take(4).ToList()));
            int EndHelix = int.Parse(ListCharToString(tmp_string_list.Skip(33).Take(4).ToList()));
            int TypeHelix = int.Parse(ListCharToString(tmp_string_list.Skip(38).Take(4).ToList()));
            int[] start_end = new int[2];
            start_end[0] = StartHelix;
            start_end[1] = EndHelix;
            try
            {
                Helix!.Add(HelixNumber, start_end);
                HelixType!.Add(HelixNumber, TypeHelix);
            }
            catch
            {

            }
        }
        public void DealSheet(string SheetString)
        {
            string[] Sheet = new string[5];
            List<char> tmp_string_list = SheetString.ToList();
            Sheet[0] = ListCharToString(tmp_string_list.Skip(11).Take(3).ToList());
            Sheet[1] = int.Parse(ListCharToString(tmp_string_list.Skip(7).Take(3).ToList())).ToString();
            Sheet[2] = ListCharToString(tmp_string_list.Skip(22).Take(4).ToList());
            Sheet[3] = ListCharToString(tmp_string_list.Skip(33).Take(4).ToList());
            Sheet[4] = int.Parse(ListCharToString(tmp_string_list.Skip(38).Take(2).ToList())).ToString();
            SheetList.Add(Sheet);
        }
        public void DealProteinString(string ProteinString)
        {
            List<string> ProteinTexts = ProteinString.Split("\n").ToList();
            for (int i = 0; i < ProteinTexts.Count(); i = i + 1)
            {
                string tmp_string = ProteinTexts[i];
                List<char> tmp_string_list = tmp_string.ToList();
                if (ListCharToString(tmp_string_list.Take(4).ToList()).Equals("ATOM"))
                {
                    try
                    {
                        DealAtoms(tmp_string);
                    }
                    catch
                    {

                    }
                }
                else if (ListCharToString(tmp_string_list.Take(5).ToList()).Equals("HELIX"))
                {
                    try
                    {
                        DealHelix(tmp_string);
                    }
                    catch
                    {

                    }
                }
                else if (ListCharToString(tmp_string_list.Take(5).ToList()).Equals("SHEET"))
                {
                    try
                    {
                        DealSheet(tmp_string);
                    }
                    catch
                    {

                    }
                }
            }
        }
        public string NodeCsv()
        {
            string R_txt = "Id,Name,x,y,z\n";
            foreach (int i in CarbenPosition.Keys)
            {
                string NodeId = i.ToString();
                string AAName = AlphaCarben[i];
                List<double> position = CarbenPosition[i];
                string x = position[0].ToString();
                string y = position[1].ToString();
                string z = position[2].ToString();
                R_txt = R_txt + NodeId + "," + AAName + "," + x + "," + y + "," + z + "\n";
            }
            return (R_txt);
        }
        public string SheetExport(string[] sheet)
        {
            string SheetID = sheet[0].Replace(" ", "");
            string SheetNum = sheet[1];
            string StartSheet = sheet[2].Replace(" ", "");
            string EndSheet = sheet[3].Replace(" ", "");
            string SheetType = sheet[4];
            string RS_Text = "";
            for (int k = int.Parse(StartSheet); k < int.Parse(EndSheet); k = k + 1)
            {
                string type = "SHEET_" + SheetID + "_" + SheetNum + "_" + SheetType;
                RS_Text = RS_Text + k.ToString() + "," + (k + 1).ToString() + "," + type + "\n";
            }
            return (RS_Text);
        }
        public string EdgeExport()
        {
            string R_Text = "src,dst,type\n";
            List<int> AAKeys = AlphaCarben.Keys.ToList();
            AAKeys.Sort();
            for (int i = 0; i < AAKeys.Count() - 1; i = i + 1)
            {
                string src = AAKeys[i].ToString();
                string drt = AAKeys[i + 1].ToString();
                string type = "CON";
                R_Text = R_Text + src + "," + drt + "," + type + "\n";
            }//C=ON
            List<int> HelixNum = Helix.Keys.ToList();
            for (int i = 0; i < HelixNum.Count(); i = i + 1)
            {
                string type = "HELIX_" + HelixNum[i].ToString() + "_" + HelixType[HelixNum[i]].ToString();
                string src = Helix[HelixNum[i]][0].ToString();
                string drt = Helix[HelixNum[i]][1].ToString();
                for (int j = int.Parse(src); j < int.Parse(drt); j = j + 1)
                {
                    R_Text = R_Text + j.ToString() + "," + (j + 1).ToString() + "," + type + "\n";
                }
            }//Helix
            for (int i = 0; i < SheetList.Count(); i = i + 1)
            {
                R_Text = R_Text + SheetExport(SheetList[i]);
            }//Sheet
            return (R_Text);
        }
        public void ExportCsv(string FilePath, string PDB_ID)
        {
            string FloderPath = Path.Combine(FilePath, PDB_ID);
            if (!Directory.Exists(FloderPath))
            {
                Directory.CreateDirectory(FloderPath);
            }
            string NodePath = Path.Combine(FloderPath, "node.csv");
            string EdgePath = Path.Combine(FloderPath, "edge.csv");
            string NodeText = NodeCsv();
            File.WriteAllText(NodePath, NodeText);
            string EdgeText = EdgeExport();
            File.WriteAllText(EdgePath, EdgeText);
        }
    }
    public class Graph_Builder
    {
        private static readonly HttpClient client = new HttpClient();
        public static async Task<string> GetPDB(string ProteinID)
        {
            client.DefaultRequestHeaders.Accept.Clear();
            string url = @"https://alphafold.ebi.ac.uk/files/AF-" + ProteinID.ToUpper() + "-F1-model_v4.pdb";
            HttpResponseMessage response = await client.GetAsync(url);
            string http_code = response.StatusCode.ToString();
            string pdb_string = "";
            if (http_code.Equals("OK"))
            {
                pdb_string = await response.Content.ReadAsStringAsync();
            }
            return (pdb_string);
        }

    }
}
namespace MainProgram
{
    public class ResultJson
    {
        public string? identifier { set; get; }
        public double score { set; get; }
    }
    public class RequestJson
    {
        public string? query_id { get; set; }
        public string? result_type { get; set; }
        public int total_count { get; set; }
        public List<ResultJson>? result_set { get; set; }

    }
    public class ConfigJson
    {
        public string? DataFolder { get; set; }
        public string? DatabaseFile { get; set; }
        public string? ProtenRecordFiles { get; set; }

    }
}