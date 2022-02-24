echo --PYTHON--
mkdir bench_python_compiled
python -m mlprodict benchmark_doc --verbose=1 --out_raw=bench_python_compiled.xslx --out_summary=bench_sum_python_compiled.xlsx --dump_dir=./bench_python_compiled --runtime=python_compiled

echo --ONNXRUNTIME--
mkdir bench_onnxruntime1
python -m mlprodict benchmark_doc --verbose=1 --out_raw=bench_onnxruntime1.xslx --out_summary=bench_sum_onnxruntime1.xlsx --dump_dir=./bench_onnxruntime1 --runtime=onnxruntime1
