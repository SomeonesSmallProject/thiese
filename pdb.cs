#define Args
#undef Args
using Protein_Graph;

namespace MainProgram
{
    public class Program
    {

#if Args
        public static void Main()
        {
            string pdb_folders = "/home/lzg/CASF-2016/coreset";
            string output_folder = "/home/lzg/transformer/output";
            foreach(string sub_folder in Directory.GetDirectories(pdb_folders))
            {
                string pdb_name = Path.GetFileName(sub_folder);
                Console.WriteLine(pdb_name);
                List<string> pdb_file_list = Directory.GetFiles(sub_folder).ToList();
                string pdb_file = Path.Combine(sub_folder,pdb_name+"_protein.pdb");
                Protein tmp_protein = new Protein();
                string pdb_txt = File.ReadAllText(pdb_file);
                tmp_protein.DealProteinString(pdb_txt);
                tmp_protein.ExportCsv(output_folder, pdb_name);
            }
        }
#else
        public static void tmpMain()
        {
            Console.WriteLine("Input_pdb_folder");
            string Input_folder = Console.ReadLine();
            if (!Directory.Exists(Input_folder))
            {
                Console.WriteLine("Pdb_folder Not exists,choose Current Directory");
                Input_folder = Directory.GetCurrentDirectory();
            }
            Console.WriteLine("Output_folder");
            string output_folder = Console.ReadLine();

            if (!Directory.Exists(output_folder))
            {
                Directory.CreateDirectory(output_folder);
            }

            List<string> pdb_file_list = Directory.GetFiles(Input_folder).ToList();
            foreach (string pdb_file in pdb_file_list)
            {
                string pdb_name = Path.GetFileNameWithoutExtension(pdb_file);
                Protein tmp_protein = new Protein();
                string pdb_txt = File.ReadAllText(pdb_file);
                tmp_protein.DealProteinString(pdb_txt);
                tmp_protein.ExportCsv(output_folder, pdb_name);
                Console.WriteLine(pdb_name);
            }
        }
#endif         
    }
}