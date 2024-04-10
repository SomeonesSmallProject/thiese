using Microsoft.AspNetCore.Mvc;
using thies.Models;
using Protein_Graph;
using System.Diagnostics;

namespace thies.Controllers
{
    public class DTI : Controller
    {
        private readonly string python_path;
        private readonly string pickle_path;
        public DTI(Dictionary<string, string> config)
        {
            this.python_path = config["python"];
            this.pickle_path = config["DTI"];
        }

        // GET: DTI
        public ActionResult Index()
        {
            return View();
        }
        [HttpPost]
        public IActionResult SubmitForm(DTI_Module DTI)
        {
            string? Uniprot_ID = DTI.UniprotId;
            string? Smiles = DTI.SMILES;
            ViewData["Uniprot_ID"] = Uniprot_ID;
            ViewData["Smiles"] = Smiles;
            if (String.IsNullOrWhiteSpace(Smiles) || String.IsNullOrWhiteSpace(Uniprot_ID))
            {
                return View();
            }
            else
            {
                string pdb_txt = Graph_Builder.GetPDB(Uniprot_ID).Result.ToString();
                Protein tmp_protein = new Protein();
                tmp_protein.DealProteinString(pdb_txt);
                DateTime currentDateTime = DateTime.Now;
                string FolderPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "tmp", "DTI", currentDateTime.ToString("yyyyMMddHHmmss"));
                tmp_protein.ExportCsv(FolderPath, Uniprot_ID);

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = this.python_path;
                psi.Arguments = "";
                using (Process process = Process.Start(psi))
                {
                    // 读取命令的输出
                    using (System.IO.StreamReader reader = process.StandardOutput)
                    {
                        ViewData["result"] = reader.ReadToEnd();
                    }
                }
                return View();
            }
        }
    }
}
