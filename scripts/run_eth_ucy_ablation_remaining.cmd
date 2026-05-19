@echo off
setlocal EnableExtensions

rem Run the redesigned ETH/UCY adaptive-prior ablation experiments from the repository root.
rem Assumption: the star conda environment has already been activated.
rem Example:
rem   scripts\run_eth_ucy_ablation_remaining.cmd
rem   scripts\run_eth_ucy_ablation_remaining.cmd univ
rem   scripts\run_eth_ucy_ablation_remaining.cmd zara1 zara2

pushd "%~dp0\.."

set "BASE_ARGS=--dataset eth5 --start_test 10 --sample_num 20 --fde_weight 0.5 --spatial_prior_mix 0.6 --spatial_sigma 2.0 --diversity_weight 0.02 --diversity_margin 0.2 --motion_gate_bias 1.0 --spatial_gate_bias -1.0"
set "DRY_RUN=0"

if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
)

if "%~1"=="" goto default_datasets
goto arg_loop

:default_datasets
call :run_dataset univ || goto failed
call :run_dataset zara1 || goto failed
call :run_dataset zara2 || goto failed
goto done

:arg_loop
if "%~1"=="" goto done
call :run_dataset "%~1" || goto failed
shift
goto arg_loop

:done
popd
echo [DONE] Remaining ETH/UCY ablation experiments finished.
exit /b 0

:failed
set "ERR=%ERRORLEVEL%"
popd
echo [FAILED] Experiment script stopped with errorlevel %ERR%.
exit /b %ERR%

:run_dataset
set "SETNAME=%~1"
call :set_dataset_args "%SETNAME%" || exit /b %ERRORLEVEL%
echo.
echo ============================================================
echo [DATASET] %SETNAME%
echo [ARGS] %DATASET_ARGS%
echo ============================================================
call :run_one "%SETNAME%" adaptive || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" adaptive_no_motion || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" adaptive_no_spatial || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" adaptive_motion_only || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" vite_only || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" motion_only || exit /b %ERRORLEVEL%
call :run_one "%SETNAME%" spatial_only || exit /b %ERRORLEVEL%
exit /b 0

:run_one
set "SETNAME=%~1"
set "MODE=%~2"
set "MODEL=ab_%SETNAME%_%MODE%"
set "OUTDIR=output\%SETNAME%\%MODEL%"

echo.
echo [TRAIN] test_set=%SETNAME% ablation=%MODE% model=%MODEL%
if "%DRY_RUN%"=="1" goto dry_run_one
mkdir "%OUTDIR%" 2>nul
if exist "%OUTDIR%\config_train.yaml" del /q "%OUTDIR%\config_train.yaml"
python trainval.py --phase train --test_set "%SETNAME%" --train_model "%MODEL%" --ablation "%MODE%" %BASE_ARGS% %DATASET_ARGS% > "%OUTDIR%\train_stdout.txt" 2>&1
if errorlevel 1 (
    echo [ERROR] Training failed: %MODEL%
    echo         See "%OUTDIR%\train_stdout.txt"
    exit /b 1
)

echo [TEST ] test_set=%SETNAME% ablation=%MODE% model=%MODEL%
if exist "%OUTDIR%\config_test.yaml" del /q "%OUTDIR%\config_test.yaml"
python trainval.py --phase test --test_set "%SETNAME%" --train_model "%MODEL%" --ablation "%MODE%" --load_model best %BASE_ARGS% %DATASET_ARGS% > "%OUTDIR%\test_stdout_best.txt" 2>&1
if errorlevel 1 (
    echo [ERROR] Testing failed: %MODEL%
    echo         See "%OUTDIR%\test_stdout_best.txt"
    exit /b 1
)

echo [OK   ] %MODEL%
exit /b 0

:dry_run_one
echo [DRY ] mkdir "%OUTDIR%" 2^>nul
echo [DRY ] if exist "%OUTDIR%\config_train.yaml" del /q "%OUTDIR%\config_train.yaml"
echo [DRY ] python trainval.py --phase train --test_set "%SETNAME%" --train_model "%MODEL%" --ablation "%MODE%" %BASE_ARGS% %DATASET_ARGS% ^> "%OUTDIR%\train_stdout.txt" 2^>^&1
echo [DRY ] if exist "%OUTDIR%\config_test.yaml" del /q "%OUTDIR%\config_test.yaml"
echo [DRY ] python trainval.py --phase test --test_set "%SETNAME%" --train_model "%MODEL%" --ablation "%MODE%" --load_model best %BASE_ARGS% %DATASET_ARGS% ^> "%OUTDIR%\test_stdout_best.txt" 2^>^&1
exit /b 0

:set_dataset_args
set "SETNAME=%~1"
if /I "%SETNAME%"=="eth" (
    set "DATASET_ARGS=--num_epochs 300 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
    exit /b 0
)
if /I "%SETNAME%"=="hotel" (
    set "DATASET_ARGS=--num_epochs 300 --learning_rate 0.0018 --router_top_p 0.6 --rt_layers 3 --num_virtual_nodes 3"
    exit /b 0
)
if /I "%SETNAME%"=="univ" (
    set "DATASET_ARGS=--num_epochs 300 --learning_rate 0.001 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
    exit /b 0
)
if /I "%SETNAME%"=="zara1" (
    set "DATASET_ARGS=--num_epochs 300 --learning_rate 0.0012 --router_top_p 0.6 --rt_layers 3 --num_virtual_nodes 3"
    exit /b 0
)
if /I "%SETNAME%"=="zara2" (
    set "DATASET_ARGS=--num_epochs 300 --learning_rate 0.0012 --router_top_p 0.5 --rt_layers 3 --num_virtual_nodes 3"
    exit /b 0
)
echo [ERROR] Unsupported ETH/UCY test_set: %SETNAME%
echo         Expected one of: eth, hotel, univ, zara1, zara2
exit /b 1
exit /b 0
