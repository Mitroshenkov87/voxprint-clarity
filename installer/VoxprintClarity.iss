; Voxprint AI Clarity online installer (Inno Setup 6).
; ISCC installer\VoxprintClarity.iss
; Start menu folder is Voxprint, shared with the other suite apps.
; /RUNTIME=skip skips the Python download. That switch is for the CI file check.
; A normal install downloads Python 3.14 and torch 2.14.1 cu130. There is no CPU runtime.

#define AppName "VoxprintClarity"
#define AppDisplayName "Voxprint AI Clarity"
#define AppInfoFile AddBackslash(SourcePath) + "..\clarity\appinfo.py"
#if FileExists(AppInfoFile) == 0
  #error clarity\appinfo.py was not found
#endif
#define AppInfoHandle FileOpen(AppInfoFile)
#define AppVersion ""
#define AppBuild ""
#define AppCodename ""
#sub ParseAppInfoLine
  #define AppInfoLine FileRead(AppInfoHandle)
  #if AppInfoLine != ""
    #if Pos("APP_VERSION", AppInfoLine) == 1
      #define public AppVersion Copy(AppInfoLine, Pos('"', AppInfoLine) + 1, Len(AppInfoLine) - Pos('"', AppInfoLine) - 1)
    #elif Pos("APP_BUILD", AppInfoLine) == 1
      #define public AppBuild Trim(Copy(AppInfoLine, Pos("=", AppInfoLine) + 1, 16))
    #elif Pos("CODENAME", AppInfoLine) == 1
      #define public AppCodename Copy(AppInfoLine, Pos('"', AppInfoLine) + 1, Len(AppInfoLine) - Pos('"', AppInfoLine) - 1)
    #endif
  #endif
#endsub
#define AppInfoI 0
#for {AppInfoI = 0; AppInfoI < 40 && !FileEof(AppInfoHandle); AppInfoI++} ParseAppInfoLine
#expr FileClose(AppInfoHandle)
#if AppVersion == ""
  #error APP_VERSION was not read
#endif
#if AppBuild == ""
  #error APP_BUILD was not read
#endif
#define PyW "{app}\runtime\Scripts\pythonw.exe"
#define Py "{app}\runtime\Scripts\python.exe"

[Setup]
AppId={{A4E8C1D2-6B70-4F3A-9E55-1C8D0A7B2E64}
AppName={#AppDisplayName}
AppVersion={#AppVersion}
AppVerName={#AppDisplayName} {#AppVersion}
AppPublisher=Voxprint
DefaultDirName={autopf}\{#AppName}
DefaultGroupName=Voxprint
DisableProgramGroupPage=yes
UsePreviousGroup=no
UninstallDisplayName={#AppDisplayName}
UninstallDisplayIcon={app}\branding\clarity.ico
SetupIconFile=..\branding\clarity.ico
OutputDir=Output
OutputBaseFilename=VoxprintClarity-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.26100
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
RuntimeStatus=Installing Python 3.14 and PyTorch for CUDA 13. Clarity has no CPU runtime.
RuntimeFailed=The Python environment could not be installed.%n%nDetails: %1

[Files]
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-gpu.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\BUILD.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\clarity\*"; DestDir: "{app}\clarity"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\locales\*"; DestDir: "{app}\locales"; Flags: ignoreversion
Source: "..\branding\*"; DestDir: "{app}\branding"; Flags: ignoreversion
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\build_info.json"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{#PyW}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\branding\clarity.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Run]
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden; Check: RuntimeInstalled

[UninstallRun]
Filename: "{#Py}"; Parameters: """{app}\main.py"" --unregister-models-user --out ""{tmp}\clarity-users.txt"""; WorkingDir: "{app}"; Flags: runhidden; RunOnceId: "unregister-clarity"; Check: RuntimeInstalled

[Code]
function CmdParam(const Name, Default: String): String;
var
  I: Integer;
  Prefix: String;
begin
  Result := Default;
  Prefix := '/' + Uppercase(Name) + '=';
  for I := 1 to ParamCount do
    if Pos(Prefix, Uppercase(ParamStr(I))) = 1 then
    begin
      Result := Copy(ParamStr(I), Length(Prefix) + 1, MaxInt);
      Exit;
    end;
end;

function RuntimeInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{#Py}'));
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Script, LogFile, Params, Skip: String;
  Rc: Integer;
  Page: TOutputMarqueeProgressWizardPage;
  Show: Integer;
begin
  Result := '';
  ExtractTemporaryFile('install-runtime.ps1');
  ExtractTemporaryFile('requirements.txt');
  ExtractTemporaryFile('runtime_lock.json');
  Script := ExpandConstant('{tmp}\install-runtime.ps1');
  LogFile := ExpandConstant('{%TEMP}\VoxprintClarity-setup.log');
  if CompareText(CmdParam('RUNTIME', 'install'), 'skip') = 0 then
    Skip := ' -SkipRuntime'
  else
    Skip := '';
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + ExpandConstant('{tmp}\requirements.txt') +
            '" -Lock "' + ExpandConstant('{tmp}\runtime_lock.json') +
            '" -Log "' + LogFile + '"' + Skip;
  if WizardSilent then Show := SW_HIDE else Show := SW_SHOWNORMAL;
  Page := CreateOutputMarqueeProgressPage('Installing', CustomMessage('RuntimeStatus'));
  Page.Show;
  try
    Page.Animate;
    if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, '', Show, ewWaitUntilTerminated, Rc) then
      Rc := -1;
  finally
    Page.Hide;
  end;
  if Rc <> 0 then
    Result := FmtMessage(CustomMessage('RuntimeFailed'), [LogFile]);
end;
