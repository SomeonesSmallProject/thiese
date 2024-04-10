using Newtonsoft.Json;
string Json_File_path = Path.Join(Directory.GetCurrentDirectory(), "./wwwroot/settings.json");
Dictionary<string, string> config = new();
if (!File.Exists(Json_File_path))
{
    return;
}
else
{
    string json_text = File.ReadAllText(Json_File_path);
    config = JsonConvert.DeserializeObject<Dictionary<string, string>>(json_text);
}

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddSingleton<Dictionary<string, string>>(config);
builder.Services.AddControllersWithViews();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
}
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
