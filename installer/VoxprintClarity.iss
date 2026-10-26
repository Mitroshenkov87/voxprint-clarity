; VOXPrint AI Clarity online installer (Inno Setup 6).
; Build the launchers first: powershell -File installer\launchers\build.ps1
; ISCC installer\VoxprintClarity.iss
; Program files live in {autopf}\VOXPrint AI Suite\Clarity.
; The Start menu folder is VOXPrint AI Suite.
; /RUNTIME=skip skips the shared runtime. That switch is for the CI file check.
; A normal install calls enlarge.suite_runtime.ensure_runtime("clarity").
; Per-user data stays in %LOCALAPPDATA%\Voxprint. This script does not delete it.

#define AppDisplayName "VOXPrint AI Clarity"
#define Publisher "VOXPrint AI Suite"
#define SuiteFolder "VOXPrint AI Suite"
#define AppFolder "Clarity"
#define GuiExe "voxprint-clarity.exe"
#define CliExe "voxprint-clarity-cli.exe"
#define OldDirName "VoxprintClarity"
#define OldMenu "Voxprint"
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
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#SuiteFolder}\{#AppFolder}
DisableDirPage=yes
UsePreviousAppDir=no
DefaultGroupName={#SuiteFolder}
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
ChangesEnvironment=yes
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
RuntimeStatus=Installing the shared suite runtime. Clarity has no CPU runtime.
RuntimeFailed=The Python environment could not be installed.%n%nDetails: %1

[Files]
Source: "install-runtime.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "install-runtime.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\requirements-gpu.txt"; DestDir: "{tmp}"; Flags: dontcopy
Source: "runtime_lock.json"; DestDir: "{tmp}"; Flags: dontcopy
Source: "pin_runtime.py"; DestDir: "{tmp}"; Flags: dontcopy
Source: "..\clarity\native_gap.py"; DestDir: "{tmp}"; Flags: dontcopy
Source: "fetch_common.py"; DestDir: "{tmp}"; Flags: dontcopy
Source: "stage_suite_runtime.py"; DestDir: "{tmp}"; Flags: dontcopy
Source: "launchers\{#GuiExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "launchers\{#CliExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-gpu.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-bench.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\BUILD.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\mdl.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\icons\suite\*"; DestDir: "{app}\assets\icons\suite"; Flags: ignoreversion
Source: "..\clarity\*"; DestDir: "{app}\clarity"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\locales\*"; DestDir: "{app}\locales"; Flags: ignoreversion
Source: "..\branding\*"; DestDir: "{app}\branding"; Flags: ignoreversion
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\build_info.json"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{app}\{#GuiExe}"; WorkingDir: "{app}"; IconFilename: "{app}\branding\clarity.ico"
Name: "{group}\{cm:UninstallProgram,{#AppDisplayName}}"; Filename: "{uninstallexe}"

[Registry]
; Per-user association. Uninstall deletes the ProgID and the extension entry.
; The ProgID stays Voxprint.Clarity.Project so an existing association still resolves.
Root: HKCU; Subkey: "Software\Classes\.vxclarity"; ValueType: string; ValueName: ""; ValueData: "Voxprint.Clarity.Project"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project"; ValueType: string; ValueName: ""; ValueData: "VOXPrint AI Clarity project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\branding\clarity.ico"
Root: HKCU; Subkey: "Software\Classes\Voxprint.Clarity.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#GuiExe}"" ""%1"""
; Suite discovery. Peers read this key. voxprint-common 0.1.1 has no helper for it.
Root: HKCU; Subkey: "Software\Voxprint\Clarity"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Voxprint\Clarity"; ValueType: string; ValueName: "Exe"; ValueData: "{#GuiExe}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Voxprint\Clarity"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Voxprint\Clarity"; ValueType: string; ValueName: "IpcName"; ValueData: ""; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Voxprint\Clarity"; ValueType: dword; ValueName: "ProtocolVersion"; ValueData: "0"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#GuiExe}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#GuiExe}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#GuiExe}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#CliExe}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#CliExe}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#CliExe}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"

[Run]
; PrepareToInstall runs before {app} exists. This step runs after the files
; are copied and creates {app}\runtime even when that earlier junction was lost.
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install-runtime.ps1"" -AppDir ""{app}"" -Log ""{tmp}\clarity-link.log"" -LinkOnly"; WorkingDir: "{app}"; StatusMsg: "Linking the shared runtime..."; Flags: runhidden waituntilterminated
Filename: "{#Py}"; Parameters: """{app}\main.py"" --install-suite"; WorkingDir: "{app}"; StatusMsg: "Checking the shared suite runtime..."; Flags: runhidden waituntilterminated; Check: RuntimeInstalled
Filename: "{#Py}"; Parameters: """{app}\main.py"" --register-models-user"; WorkingDir: "{app}"; StatusMsg: "Registering the shared models folder..."; Flags: runhidden waituntilterminated; Check: RuntimeInstalled

[UninstallRun]
Filename: "{#Py}"; Parameters: """{app}\main.py"" --unregister-models-user --out ""{tmp}\clarity-users.txt"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "unregister-clarity"; Check: RuntimeInstalled

[Code]
const
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{A4E8C1D2-6B70-4F3A-9E55-1C8D0A7B2E64}_is1';

var
  PreviousInstall: String;

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

function SameDir(const A, B: String): Boolean;
begin
  Result := CompareText(RemoveBackslash(A), RemoveBackslash(B)) = 0;
end;

function UnderLocalAppData(const Dir: String): Boolean;
var
  Local: String;
begin
  Local := RemoveBackslash(ExpandConstant('{%LOCALAPPDATA}'));
  Result := (Local <> '') and (Pos(Uppercase(AddBackslash(Local)), Uppercase(AddBackslash(Dir))) = 1);
end;

function ReadInstallLocation(const RootKey: Integer; var Location: String): Boolean;
begin
  Result := RegQueryStringValue(RootKey, UninstallKey, 'InstallLocation', Location);
end;

function InitializeSetup(): Boolean;
var
  Location: String;
begin
  Result := True;
  PreviousInstall := '';
  if ReadInstallLocation(HKLM, Location) or ReadInstallLocation(HKCU, Location) then
    PreviousInstall := Location;
end;

procedure DropJunction(const Path: String);
var
  Rc: Integer;
begin
  if DirExists(Path) then
    Exec(ExpandConstant('{cmd}'), '/c rmdir "' + Path + '"', '', SW_HIDE, ewWaitUntilTerminated, Rc);
end;

procedure DeleteInstallTree(const Dir: String);
var
  FindRec: TFindRec;
  Path: String;
begin
  if not DirExists(Dir) then
    Exit;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          Path := AddBackslash(Dir) + FindRec.Name;
          if (FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          begin
            if CompareText(FindRec.Name, 'runtime') <> 0 then
              DeleteInstallTree(Path);
          end
          else
            DeleteFile(Path);
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
  RemoveDir(Dir);
end;

procedure RemoveOldProgramDir(const Dir: String);
begin
  if Dir = '' then
    Exit;
  if SameDir(Dir, ExpandConstant('{app}')) then
    Exit;
  if UnderLocalAppData(Dir) then
    Exit;
  DropJunction(AddBackslash(Dir) + 'runtime');
  DeleteInstallTree(Dir);
end;

procedure RemoveShortcut(const Folder, Name: String);
begin
  if FileExists(AddBackslash(Folder) + Name) then
    DeleteFile(AddBackslash(Folder) + Name);
end;

procedure RemoveDirIfEmpty(const Folder: String);
var
  FindRec: TFindRec;
  Empty: Boolean;
begin
  if not DirExists(Folder) then
    Exit;
  Empty := True;
  if FindFirst(AddBackslash(Folder) + '*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
          Empty := False;
      until (not Empty) or (not FindNext(FindRec));
    finally
      FindClose(FindRec);
    end;
  end;
  if Empty then
    RemoveDir(Folder);
end;

procedure RemoveOldShortcuts();
begin
  RemoveShortcut(ExpandConstant('{commonprograms}\{#OldMenu}'), 'Voxprint AI Clarity.lnk');
  RemoveShortcut(ExpandConstant('{commonprograms}\{#OldMenu}'), 'Uninstall Voxprint AI Clarity.lnk');
  RemoveShortcut(ExpandConstant('{userprograms}\{#OldMenu}'), 'Voxprint AI Clarity.lnk');
  RemoveShortcut(ExpandConstant('{userprograms}\{#OldMenu}'), 'Uninstall Voxprint AI Clarity.lnk');
  RemoveDirIfEmpty(ExpandConstant('{commonprograms}\{#OldMenu}'));
  RemoveDirIfEmpty(ExpandConstant('{userprograms}\{#OldMenu}'));
end;

function RewritePath(const Orig, Drop, Add: String): String;
var
  Rest, Piece: String;
  Skip: Boolean;
begin
  Result := '';
  Rest := Orig;
  while Rest <> '' do
  begin
    if Pos(';', Rest) = 0 then
    begin
      Piece := Rest;
      Rest := '';
    end
    else
    begin
      Piece := Copy(Rest, 1, Pos(';', Rest) - 1);
      Rest := Copy(Rest, Pos(';', Rest) + 1, MaxInt);
    end;
    Piece := Trim(Piece);
    Skip := (Piece = '');
    if (not Skip) and (Drop <> '') and (CompareText(RemoveBackslash(Piece), RemoveBackslash(Drop)) = 0) then
      Skip := True;
    if (not Skip) and (Add <> '') and (CompareText(RemoveBackslash(Piece), RemoveBackslash(Add)) = 0) then
      Skip := True;
    if not Skip then
    begin
      if Result <> '' then
        Result := Result + ';';
      Result := Result + Piece;
    end;
  end;
  if Add <> '' then
  begin
    if Result <> '' then
      Result := Result + ';';
    Result := Result + Add;
  end;
end;

procedure WriteUserPath(const Drop, Add: String);
var
  Orig, Updated: String;
begin
  if (Drop = '') and (Add = '') then
    Exit;
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Orig) then
    Orig := '';
  Updated := RewritePath(Orig, Drop, Add);
  if Updated <> Orig then
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Updated);
end;

procedure MigrateOldInstall();
var
  OldDefault: String;
begin
  RemoveOldShortcuts();
  OldDefault := ExpandConstant('{autopf}\{#OldDirName}');
  RemoveOldProgramDir(OldDefault);
  if not SameDir(PreviousInstall, OldDefault) then
    RemoveOldProgramDir(PreviousInstall);
  WriteUserPath(OldDefault, '');
  if not SameDir(PreviousInstall, OldDefault) then
    WriteUserPath(PreviousInstall, '');
  WriteUserPath('', ExpandConstant('{app}'));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    MigrateOldInstall();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    WriteUserPath(ExpandConstant('{app}'), '');
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
  ExtractTemporaryFile('requirements-gpu.txt');
  ExtractTemporaryFile('runtime_lock.json');
  ExtractTemporaryFile('pin_runtime.py');
  ExtractTemporaryFile('native_gap.py');
  ExtractTemporaryFile('fetch_common.py');
  ExtractTemporaryFile('stage_suite_runtime.py');
  Script := ExpandConstant('{tmp}\install-runtime.ps1');
  LogFile := ExpandConstant('{%TEMP}\voxprint-clarity-setup.log');
  if CompareText(CmdParam('RUNTIME', 'install'), 'skip') = 0 then
    Skip := ' -SkipRuntime'
  else
    Skip := '';
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" -AppDir "' + ExpandConstant('{app}') +
            '" -Requirements "' + ExpandConstant('{tmp}\requirements.txt') +
            '" -GpuRequirements "' + ExpandConstant('{tmp}\requirements-gpu.txt') +
            '" -Lock "' + ExpandConstant('{tmp}\runtime_lock.json') +
            '" -PinScript "' + ExpandConstant('{tmp}\pin_runtime.py') +
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
