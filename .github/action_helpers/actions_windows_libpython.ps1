
param (
   [string]$pythonver = "3.8"
)

$a = $pythonver.substring(0,1)
$b = $pythonver.substring(2,1)
echo "$a$b"
echo "set-output name=python_ver::$a$b"
echo "::set-output name=python_ver::$a$b"

which python;
mkdir D:\a\python$pythonver;

Copy-Item -Path $env:pythonLocation\libs -Destination "D:\a\python$pythonver" -Recurse;
Copy-Item -Path $env:pythonLocation\include -Destination "D:\a\python$pythonver" -Recurse;
dir D:\a\python$pythonver;
echo $env:pythonLocation;
dir $env:pythonLocation\include;
Remove-Item -Recurse -Force %GITHUB_ACTION_WIN_PROJECT%\boost\stage -ErrorAction Ignore;