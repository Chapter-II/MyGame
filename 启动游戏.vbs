Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = dir
WshShell.Run Chr(34) & dir & "\.venv\Scripts\pythonw.exe" & Chr(34) & " -m mygame", 0, False
