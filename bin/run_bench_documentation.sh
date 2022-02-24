echo --PYTHON--
mkdir bench_python_compiled
python -m mlprodict validate_runtime -se 1 --verbose=1 --out_raw=bench_python_compiled.csv --out_summary=bench_sum_python_compiled.xlsx --benchmark=1 --dump_folder=./bench_python_compiled --runtime=python_compiled

echo --ONNXRUNTIME--
mkdir bench_onnxruntime1
python -m mlprodict validate_runtime -se 1 --verbose=1 --out_raw=bench_onnxruntime1.csv --out_summary=bench_sum_onnxruntime1.xlsx --benchmark=1 --dump_folder=./bench_onnxruntime1 --runtime=onnxruntime1

