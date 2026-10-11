; Voxprint AI Clarity online installer (Inno Setup 6).
; ISCC installer\VoxprintClarity.iss
; Start menu folder is Voxprint, shared with the other suite apps.
; /RUNTIME=skip skips the shared runtime. That switch is for the CI file check.
; A normal install calls voxprint_engine.suite_runtime.ensure_runtime for holder clarity.

#define AppName "VoxprintClarity"
#define AppDisplayName "Voxprint AI Clarity"
; Version, build, and codename are one key per line in BUILD.json.
#define BuildFile AddBackslash(SourcePath) + "..\BUILD.json"
#if FileExists(BuildFile) == 0
  #error BUILD.json was not found
#endif
#define BuildHandle FileOpen(BuildFile)
#define AppVersion ""
#define AppBuild ""
#define AppCodename ""
#sub ParseBuildLine
  #define BuildLine FileRead(BuildHandle)
  #if Pos('"version"', BuildLine) > 0
    #define Tail Copy(BuildLine, Pos(":", BuildLine) + 1, 200)
    #define Quoted Copy(Tail, Pos('"', Tail) + 1, 200)
    #define public AppVersion Copy(Quoted, 1, Pos('"', Quoted) - 1)
  #elif Pos('"build"', BuildLine) > 0
    #define Tail Copy(BuildLine, Pos(":", BuildLine) + 1, 32)
    #define public AppBuild Trim(StringChange(StringChange(StringChange(Tail, ",", ""), "}", ""), " ", ""))
  #elif Pos('"codename"', BuildLine) > 0
    #define Tail Copy(BuildLine, Pos(":", BuildLine) + 1, 200)
    #define Quoted Copy(Tail, Pos('"', Tail) + 1, 200)
    #define public AppCodename Copy(Quoted, 1, Pos('"', Quoted) - 1)
  #endif
#endsub
#define BuildI 0
#for {BuildI = 0; BuildI < 40 && !FileEof(BuildHandle); BuildI++} ParseBuildLine
#expr FileClose(BuildHandle)
#if AppVersion == ""
  #error version was not read from BUILD.json
#endif
#if AppBuild == ""
  #error build was not read from BUILD.json
#endif
#if AppCodename == ""
  #error codename was not read from BUILD.json
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
RuntimeStatus=Installing the shared suite runtime. Clarity has no CPU runtime.
RuntimeFailed=The Python environment could not be installed.%n%nDetails: %1

[Files]
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "install-runtime.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-gpu.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-bench.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\BUILD.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\icons\suite\*"; DestDir: "{app}\assets\icons\suite"; Flags: ignoreversion
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

[Registry]
; Per-user association. Uninstall deletes the ProgID and the extension entry.
Root: HKCU; Subkey: "Software\Classes\.vxclarity"; ValueType: string; ValueName: ""; ValueData: "Voxprint.Clarity.Project"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project"; ValueType: string; ValueName: ""; ValueData: "Voxprint AI Clarity project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\branding\clarity.ico"
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{#PyW}"" ""{app}\main.py"" ""%1"""

[Run]
; PrepareToInstall runs before {app} exists. This step runs after the files
; are copied and creates {app}\runtime even when that earlier junction was lost.
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install-runtime.ps1"" -AppDir ""{app}"" -Log ""{tmp}\clarity-link.log"" -LinkOnly"; WorkingDir: "{app}"; StatusMsg: "Linking the shared runtime..."; Flags: runhidden waituntilterminated
Filename: "{#Py}"; Parameters: """{app}\main.py"" --install-suite"; WorkingDir: "{app}"; StatusMsg: "Checking the shared suite runtime..."; Flags: runhidden waituntilterminated; Check: RuntimeInstalled
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden waituntilterminated; Check: RuntimeInstalled

[UninstallRun]
Filename: "{#Py}"; Parameters: """{app}\main.py"" --unregister-models-user --out ""{tmp}\clarity-users.txt"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "unregister-clarity"; Check: RuntimeInstalled

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
