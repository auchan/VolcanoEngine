set compiler="D:\Program Files\VulkanSDK\1.1.114.0\Bin\glslc.exe"
@echo off
for /r %%i in (*.vert) do %compiler% %%i -o bin/%%~ni_vert.spv
for /r %%i in (*.frag) do %compiler% %%i -o bin/%%~ni_frag.spv
pause