using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

internal static class Program
{
    private const string DisplayName = "VOXPrint AI Clarity";

#if CLI
    private const string PythonName = "python.exe";
#else
    private const string PythonName = "pythonw.exe";

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr owner, string text, string caption, uint type);
#endif

    private static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar);
        string python = Path.Combine(dir, "runtime", "Scripts", PythonName);
        string script = Path.Combine(dir, "main.py");
        if (!File.Exists(python) || !File.Exists(script))
        {
            Fail("The shared runtime is not linked. Run the installer again.");
            return 2;
        }

        var start = new ProcessStartInfo();
        start.FileName = python;
        start.Arguments = Quote(script) + Join(args);
        start.WorkingDirectory = dir;
        start.UseShellExecute = false;
        try
        {
            Process process = Process.Start(start);
            if (process == null)
            {
                Fail("The shared runtime could not be started.");
                return 2;
            }
            process.WaitForExit();
            return process.ExitCode;
        }
        catch (Exception ex)
        {
            Fail(ex.Message);
            return 2;
        }
    }

    private static void Fail(string message)
    {
#if CLI
        Console.Error.WriteLine(DisplayName + ": " + message);
#else
        MessageBox(IntPtr.Zero, message, DisplayName, 0x10);
#endif
    }

    private static string Join(string[] args)
    {
        var builder = new StringBuilder();
        for (int i = 0; i < args.Length; i++)
        {
            builder.Append(' ');
            builder.Append(Quote(args[i]));
        }
        return builder.ToString();
    }

    private static string Quote(string value)
    {
        if (value.IndexOf(' ') < 0 && value.IndexOf('\t') < 0 && value.IndexOf('"') < 0)
        {
            return value;
        }
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }
}
