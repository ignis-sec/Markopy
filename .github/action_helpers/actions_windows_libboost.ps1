
param (
    [string]$pythonver = "3.8"
)

$a = $pythonver.substring(0,1)
$b = $pythonver.substring(2,1)
echo "$a$b"

echo $env:RUNNER_WORKSPACE
mkdir $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\lib
dir  $env:GITHUB_ACTION_WIN_PROJECT\boost;
cp $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\x64\Release\lib\boost_program_options.dll $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\lib\boost_program_options-vc142-mt-x64-1_76.dll;
cp $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\x64\Release\lib\boost_program_options.lib $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\lib\boost_program_options-vc142-mt-x64-1_76.lib;

cp $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\x64\Release\lib\boost_python$a$b.dll $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\lib\libboost_python$a$b-vc142-mt-x64-1_76.dll;
cp $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\x64\Release\lib\boost_python$a$b.lib $env:GITHUB_ACTION_WIN_PROJECT\boost\stage\lib\libboost_python$a$b-vc142-mt-x64-1_76.lib;
