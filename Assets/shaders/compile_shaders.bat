
set compiler="%VULKAN_SDK%\Bin\glslc.exe"
@set compiler_workdir=%~dp0
set compiler_workdir=%compiler_workdir:~0,-1%

@echo off
pushd .
cd %compiler_workdir%
if not exist bin (
    mkdir bin
)

set has_compile_error=0
setlocal EnableDelayedExpansion 
for /r %%i in (*.vert) do (
    %compiler% %%i -o bin/%%~ni_vert.spv
    if !ERRORLEVEL! NEQ 0 (
        set has_compile_error=1
    )
)
for /r %%i in (*.frag) do (
    %compiler% %%i -o bin/%%~ni_frag.spv
    if !ERRORLEVEL! NEQ 0 (
        set has_compile_error=1
    )
)

setlocal DisableDelayedExpansion
popd
exit /b %has_compile_error%
