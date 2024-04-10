#define DEBUG
#undef DEBUG
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using System.Diagnostics;
using thies.Models;

namespace thies.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class Mol : Controller
    {
        private readonly Dictionary<string, string> _modelProperties = new();
        private readonly string python_path;

        public Mol(Dictionary<string, string> config)
        {
            this._modelProperties["ESOL"] = config["ESOL"];
            this._modelProperties["Lipophilicity"] = config["Lipophilicity"];
            this.python_path = config["python"];
        }
        // GET: Mol
        public ActionResult Index()
        {
            ViewData["ModelProperties"] = _modelProperties;
            return View();
        }

        [HttpPost]
        public IActionResult SubmitForm(Mol_Module mol)
        {
            string? Module = mol.Selected.ToString();
            string? filePath = "";
            if (!string.IsNullOrEmpty(mol.drawed))
            {
                string? mol_text = System.Web.HttpUtility.UrlDecode(mol.drawed.ToString());
                DateTime currentDateTime = DateTime.Now;
                filePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot","tmp","mol", $"{currentDateTime.ToString("yyyyMMddHHmmss")}.mol2");
                StreamWriter ws = new StreamWriter(filePath);
                ws.Write(mol_text);
            }
            string? SMILES = mol.SMILES;
            IFormFile? file = mol.File;
            if (file != null && file.Length > 0)
            {
                var fileName = Path.GetFileName(file.FileName);
                filePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "tmp", "mol", fileName);

                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    file.CopyTo(stream);
                }
            }
            List<string> svgImages = new();
#if DEBUG
#else
            if (String.IsNullOrWhiteSpace(SMILES) & String.IsNullOrWhiteSpace(filePath))
            {
                return View();
            }
            else
            {
                string python_pickle = this._modelProperties[Module];
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = this.python_path;
                string script_path = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "python", "Mol", "Mol.py");
                psi.Arguments = $" {script_path} --pickle={python_pickle} --smiles={SMILES}";
                psi.RedirectStandardOutput = true; // 重定向标准输出
                psi.UseShellExecute = false; // 必须为false才能重定向输出流
                psi.CreateNoWindow = true; // 不创建新窗口
                string result = "";
                using (Process process = Process.Start(psi))
                {
                    process.WaitForExit();
                    // 读取命令的输出
                    using (System.IO.StreamReader reader = process.StandardOutput)
                    {
                        result = reader.ReadToEnd();
                    }
                }
                var html_result = JsonConvert.DeserializeObject<Dictionary<string, object>>(result);
                ViewData["pre"] = html_result[$"{SMILES}_score"];
                svgImages = new List<string>
                {
                    html_result[$"{SMILES}_image"].ToString(),
                }
                ;
            }
                
#endif
                ViewData["svgs"] = svgImages;
                return View();
            
        }
    }
}
